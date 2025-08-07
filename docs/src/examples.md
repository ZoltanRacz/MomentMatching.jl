```@meta
    ShareDefaultModule = true
```

# Example: Estimating an AR(1) process with noise

Consider a stochastic process that is a sum of an AR(1) process and a white noise as follows:

```math
\begin{align*}
y_{i,t} &= z_{i,t} + \nu_{i,t}\\
z_{i,t} &= \rho \cdot z_{i,t-1} + \varepsilon_{i,t},
\end{align*}
```
where
```math
 \varepsilon_{i,t} \sim \mathcal{N}(0,\sigma_\varepsilon^2) \qquad  \nu_{i,t} \sim \mathcal{N}(0,\sigma_\nu^2)
```
are i.i.d. shocks. The aim is to estimate parameters ``(\rho, \sigma_\varepsilon, \sigma_\nu)`` based on a set of moments ``\text{Var}(y_{t}), \text{Cov}(y_{t}, y_{t-1}), \text{Cov}(y_{t}, y_{t-2})`` computed from an observed sample of ``y_{i,t}``s.

## Setting up the problem

Being about to run a structural estimation exercise, one usually already has a piece of code that given
 - a guess for the estimated parameters;
 - (possibly) a set of other parameters that are treated as fixed;
 - (possibly) a draw of random shocks;
 - (possibly) a set of empty arrays (to avoid allocation while evaluating the function)

can compute a set of counterfactual moments, which can then be compared to their empirical counterpart.

To utilize this package one has to wrap this objective function and its inputs into functions and abstract types defined within the package. This is shown below.

First, one needs to define an estimation mode:

```@example
using MomentMatching # hide
struct AR1Estimation <: EstimationMode 
    "mode-dependent prefix of filenames used for saving estimation results"
    filename::String
end
```

During the estimation, one needs to evaluate the objective function for each parameter guess. Passing invariant parameters to the objective function is possible via defining an auxiliary structure which has to be a subtype of `AuxiliaryParameters`. One also needs to write a corresponding function to generate a default auxiliary structure as shown below. In this case, we pass the dimensions of the simulated sample.

```@example
struct AR1AuxPar{T<:Integer} <: AuxiliaryParameters
    "sample size of simulation"
    Nsim::T
    "number of time periods to simulate"
    Tsim::T
    "number of periods to discard for moment evaluation "
    Tdis::T
end

AuxiliaryParameters(mode::AR1Estimation, modelname::String) = AR1AuxPar(10000, 200, 100)

nothing # hide
```

It is crucial that the same set of shocks are used during the parameter estimation, as otherwise convergence cannot be achieved in the local minimization phase (the sensitivity of results to different draws of shocks can be checked via bootstrapping, as explained later in the [`Inference`](@ref Example.Inference) section). This is again done by defining an appropriate subtype of an existing abstract type and a function generating a default container of shocks. In this case, one needs to draw a normal shock for ``\varepsilon`` and ``\nu`` for each `t` and `n`.

```@example
struct AR1PreShocks{S<:AbstractFloat} <: PredrawnShocks
    "preallocated array for persistent shocks"
    ϵs::Array{S,2}
    "preallocated array for transitory shocks"
    νs::Array{S,2}
end

function PredrawnShocks(mode::AR1Estimation, modelname::String, typemom::String,aux::AuxiliaryParameters)
    return AR1PreShocks(randn(aux.Nsim, aux.Tsim),
        randn(aux.Nsim, aux.Tsim))
end

nothing # hide
```

In order to compute the necessary moments of large samples, one often needs to populate large arrays with realized values (in our case, of ``y_{i,t}``s). Creating separate containers for each guess for the parameter vector would be very costly, so instead this is done once before starting the estimation, and the data contained within will be repeatedly overwritten. 

!!! note 
    When performing an estimation via parallel computing, these containers are internally generated separately for each thread, and hence data race is automatically avoided.

In this example, we will compute cross-sectional moments in each time period and take their time-average in the final step. Therefore, we need to keep track of ``z`` and ``y`` (together its first and second lags) and the already computed moments. Defining the structure of preallocated data follows a similar logic as the previous steps.

```@example
struct AR1PrealCont{S<:AbstractFloat} <: PreallocatedContainers
    z::Vector{S}
    y::Vector{S}
    ylag1::Vector{S}
    ylag2::Vector{S}
    mat::Array{S,2}
end

function PreallocatedContainers(mode::AR1Estimation, modelname::String, typemom::String,
 aux::AuxiliaryParameters)

    z = Vector{Float64}(undef, aux.Nsim)
    y = Vector{Float64}(undef, aux.Nsim)
    ylag1 = Vector{Float64}(undef, aux.Nsim)
    ylag2 = Vector{Float64}(undef, aux.Nsim)

    mat = Array{Float64}(undef, 3, aux.Tsim) # one row for each moment

    return AR1PrealCont(z, y, ylag1, ylag2, mat)
end
```


Now we are in the position of constructing the objective function. This is done via writing a method for `MomentMatching.obj_mom!`, specializing it on the subtype `AR1Estimation` created before.

```@example
using Statistics
function MomentMatching.obj_mom!(mom::AbstractVector, momnorm::AbstractVector,
 mode::AR1Estimation, x::Array{Float64,1}, modelname::String, typemom::String,
  aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers;
   saving_model::Bool=false, filename::String="")
    (ρ, σϵ, σν) = x

    for n in 1:aux.Nsim
        preal.z[n] = 0.0
    end
    for t in 1:aux.Tsim
        for n in 1:aux.Nsim
            preal.z[n] = ρ * preal.z[n] + σϵ * presh.ϵs[n, t]
            preal.y[n] = preal.z[n] + σν * presh.νs[n, t]
        end
        if t > 2
            preal.mat[3, t] = cov(preal.y, preal.ylag2)
            copy!(preal.ylag2, preal.ylag1)
        end
        if t > 1
            preal.mat[2, t] = cov(preal.y, preal.ylag1)
            copy!(preal.ylag1, preal.y)
        end
        preal.mat[1, t] = var(preal.y)
        copy!(preal.ylag1, preal.y)
    end

    mom[1] = mean(@view preal.mat[1, aux.Tdis:end])
    momnorm[1] = mom[1]

    mom[2] = mean(@view preal.mat[2, aux.Tdis:end])
    momnorm[2] = mom[2]

    mom[3] = mean(@view preal.mat[3, aux.Tdis:end])
    momnorm[3] = mom[3]

end
```

We give the names and ranges of the targeted parameters by writing a method of `parambounds`. During the global phase of the estimation, the region within 'global' bounds is searched. Violating 'hard' bounds during the local phase induces a penalty to redirect the algorithm towards the allowed range.

```@example
function MomentMatching.parambounds(mode::AR1Estimation)
    full_labels    = [ "ρ",  "σϵ",  "σν"]
    full_lb_hard   = [ 0.0,   0.0,  0.0 ]
    full_lb_global = [ 0.0,   0.0,  0.0 ]
    full_ub_global = [ 1.0,   1.0,  1.0 ]
    full_ub_hard   = [ 1.0,   Inf,  Inf ]
    return full_labels, full_lb_hard, full_lb_global, full_ub_global, full_ub_hard
end
```

Next, we specify which moments are targeted during the estimation. In an actual application, this function would most likely read in values from a dataset, but here we just give three arbitrary numbers for each moments.

```@example
function MomentMatching.datamoments(mode::AR1Estimation, typemom::String)
    momtrue = [0.8, 0.45, 0.4] # made up numbers

    mmomtrue = deepcopy(momtrue)

    return hcat(momtrue, mmomtrue)
end
```

Finally, we name the targeted moments. The `momentnames` function has to return a `DataFrame` with two columns, where one targeted moment corresponds to one row. If the two moments have coinciding values in the first column, the corresponding results will be visualized together, as shown in section [`Estimation`](@ref Example.Estimation).

```@example
using DataFrames
function MomentMatching.momentnames(mode::AR1Estimation, typemom::String)
    moments = fill("Cov(y_t,y_t-j)", 3)
    lags = string.(0:2)
    return DataFrame(Moment=moments, Lags=lags)
end
```

!!! note 
    By default the deviation between data and model moments is obtained by rescaling the difference between the two with data means of each moment. For instance, if one targets the time-series of the cross-sectional skewness of the distribution of income growth, the differences in each year would be scaled by the time-series average of the cross-sectional skewness (clearly, if one targets just one moment the mean is the moment itself). The user can change this by writing a mode-specific `mdiff` function (see related code in `estimation.jl`).

To summarize, when applying this package for existing code, follow these steps:

1. Set up an `EstimationMode` structure.
2. Write `EstimationMode`-specific auxiliary structures `AuxiliaryParameters` (for any input that stays constant all over the estimation), `PredrawnShocks` (for simulation noise) and `PreallocatedContainers` (containers which can be overwritten over the estimation) whenever relevant. These will be inputs of the objective function.
3. Set the bounds of the parameter space and the values of the moments to be matched by writing an `EstimationMode`-specific `MomentMatching.parambounds` and `MomentMatching.datamoments` functions. 
4. Wrap the objective function within an appropriate method of `MomentMatching.obj_mom!`.

## [Estimation](@id Example.Estimation)
After defining an estimation setup (see section [`Estimating alternative specifications`](@ref Example.Alternative) for more details on why this structure is useful) and a structure supplying numerical settings, one can perform the estimation as follows. After checking 100 points in the global phase, a local minimization takes place using the Nelder-Mead algorithm, started from the 10 global points with the lowest objective function values. 

!!! note 
    In this example we use the default weighting matrix - which is the unitary matrix - but the user can change this by defining a a mode-specific `default_weight_matrix` function (see related code in `estimation.jl`) or by passing their preferred weighting matrix via the keyword argument `Wmat` to `estimation`.

```@example
using OptimizationOptimJL
setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npest = NumParMM(setup; Nglo=100, Nloc=10,
 local_opt_settings = (algorithm = NelderMead(), maxtime = 30.0))

est = estimation(setup; npmm=npest, saving=false); 
nothing # hide
```

The estimated parameters can be displayed as follows:
```@example
tableest(setup, est)
```

The match with targeted moments can either be displayed as a table
```@example
tablemoms(setup, est)
```
or visualized on a figure:
```@example
using Plots
fmoms(setup, est)
savefig("fmoms.svg"); nothing # hide
```
![](fmoms.svg)

As in this case 3 parameters were estimated based on 3 moments (and hence parameters are exactly identified), the resulting match is very close.

Results can be saved by setting `saving` equal to `true`. In this case `filename` specified in estimation mode will be used as suffix. The default saving path is `"./saved/estimation_results/"`. 

## Diagnostics

When the objective function is highly non-linear, it is in general difficult to know if the obtained parameter estimate indeed corresponds to a global minimizer. One concern would be that the obtained local optimum is 'too local', i.e. its basin of attraction is too narrow. In this case the local optimum would be very sensitive to the respective initial point. To judge the accuracy of the estimated parameter vector, two heuristic methods are available in this package.

First, it is possible to visualize how the objective function depends on varying the parameter estimates one-at-a-time (keeping the other parameters constant), around the best point.

```@example
marg = marginal_fobj(setup, est, 17, fill(0.1, 3))
fmarg(setup, est, marg)
savefig("fmarg.svg"); nothing # hide
```
![](fmarg.svg)

Second, one can visualize how sensitive the corresponding parameter values are to the rank of the corresponding global or local point, with respect to their objective function values. This is informative on the sufficient number of global and local points.

Output from the global stage is available via the `global` keyword.

```@example
fsanity(setup, est, glob = true)
savefig("fsanity_glo.svg"); nothing # hide
```
![](fsanity_glo.svg)

By default, results from the local stage are shown.

```@example
fsanity(setup, est)
savefig("fsanity_loc.svg"); nothing # hide
```
![](fsanity_loc.svg)

## [Inference](@id Example.Inference)

### Parametric Bootstrap

Even if the model is correctly specified, there are two reasons why parameters are estimated with an error:
1. The targeted population moments are obtained from a finite sample.
2. If evaluating the objecting function involves uncertainty, the whole estimation procedure is conducted with one particular draw of shocks. This makes results potentially sensitive to this specific realization of shocks.

One can gauge the joint effect of these forces on the precision of the estimates via parametric bootstrapping. 
 - First, using the obtained parameter estimates, ``N_{sample}`` independent samples are created to mimic the uncertainty in the data generating process. The targeted moments are then computed from each of these samples. Note that the size of the simulated samples have to coincide with the actual data sample which was used to compute the data moments.
  - Second, if computing the objective function involves random draws, ``N_{seed}`` number of different shocks are draws.

Then for each pair of alternative moments and seeds, the local stage of the estimation is repeated starting from the best local point of the original estimation. The distribution of the resulting ``N_{sample} \cdot N_{seed}`` new estimates can then be used to generate confidence intervals.

```@example
Tdis = 20 # burn in
Tdata = 40 # true data length
Ndata = 500 # true sample size
Nsample = 15 # number of samples used for bootstrap
Nseed = 15 # number of shock simulations used for bootstrap
auxmomsim = AR1AuxPar(Ndata, Tdata + Tdis, Tdis)
boot = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false);

fbootstrap(setup, est, boot)
savefig("fbootstrap.svg"); nothing # hide
```
![](fbootstrap.svg)

## [Multithreading and multiprocessing](@id Example.Multi)
The global and local phases of the estimation procedure require evaluating the objective function at many points of the parameter space. In our package this task can be parallelized with multithreading (`Threads` module, distributes across cores within a process), multiprocessing (`Distributed` module, distributes across different processes) and/or a combination of the two (distributes across different processes and then across the cores within a process) by appropriately setting the structure `ComputationSettings`. To apply some given computational settings, one just need to pass it to the `estimation` function with the keyword argument `cs`. We describe below how to do this locally on one's computer and on a cluster.

### Local parallelization

#### Multithreading

In this case the only relevant field in `ComputationSettings` is `num_tasks`. If for example we run
```@example
cs_1 = ComputationSettings(num_tasks=4)
est_1 = estimation(setup; npmm=npest, cs=cs_1, saving=false);
nothing #hide
```
then in the global estimation phase, four tasks will be spawn, with each containing a quarter of all points where the objective function needs to be computed. These tasks are then queued at the available threads, the number of which depends on how Julia was started (as usual), in particular the option `-t/--threads` command line argument or  the `JULIA_NUM_THREADS` environment variable (see [here](https://docs.julialang.org/en/v1/manual/multi-threading/)). After the global phase is finished, the starting points for the local phase are also shared between 4 tasks.

!!! note
    - Choosing a higher number for `num_tasks` than `JULIA_NUM_THREADS` and thus overscheduling the threads might help prevent idleness. 
    - The default value for `num_tasks` is `Threads.nthreads()*2`, implying that multithreading is **active by default** and that threads are overscheduled.
    - Forcing single-threading is possible via setting `num_tasks` to 1.
    
!!! note
    When using multithreading, separate `PreallocatedContainers` are spawn for each task, and the containers of estimation outputs are accessed elementwise, hence data races are prevented as long as no other object is being overwritten in the objective function written by the user.


#### Multiprocessing

Given that memory is not shared across the different processes, before running any code using multiprocessing we need to make sure that the required elements (functions, packages, structures, types...) are loaded in each of them. The function to do that is `load_on_procs`. Specifically, one writes a Julia script dedicated to loading all the required elements and calls it in `load_on_procs` which takes care of running it in every process. In our case such file is called `minimalAR1.jl` and it basically loads the functions, packages, structures, types, etc. that we have used so far in this example (if you want to have a look, the script is available in the `test/examples` folder of the GitHub repository of the package). Be sure to specify the path correctly when calling `include`.

```julia-repl
julia> using Distributed
julia> function MomentMatching.load_on_procs(mode::AR1Estimation)
    return @everywhere begin include("minimalAR1.jl") end
end
```

Besides telling Julia what to run on each process when starting them, we also need to specify
 * how many processes we want
 * with how many threads the new Julia instances should start
 
via the `num_proc` and `num_threads` fields of the `ComputationalSettings` structure. Note that when multiprocessing is active, `num_tasks` will set the number of tasks per process.

For example,
 - `ComputationSettings(num_procs=8, num_threads=1, num_tasks=1)` starts 8 processes without any multithreading on each. At most 8 points can be evaluated at the same time.
 - `ComputationSettings(num_procs=8, num_threads=4, num_tasks=8)` starts 8 processes with 4 threads and 8 tasks on each. On each process, 4 tasks can immediately be scheduled to threads immediately, while 4 other ones will wait. At most 32 points can be evaluated at the same time.

!!! note
    When via the `estimation` function both the global and local phases are performed in one go, the same specified computational settings are applied to both. It is also possible to run each phase separately with its own computational settings (see the section [`Only global or only local`](@ref Example.Onlyone) the description of how to run the two stages separately). `ComputationSettings` also works in the function performing bootstrapping.

!!! danger 
    While the package is designed in a way to prevent data races in the estimation and bootstrapping routines, it is always the user's responsibility to check that this does not happen in their own model. In addition, it is best to avoid applying multithreading in one's model code within `obj_mom!` if multithreading is already active when looping over points in the parameter space.

### Parallelization on a cluster
Currently, our package works only on clusters using Slurm Workload Manager. This is an example on how to set `ComputationSettings` for running the estimation on Slurm:
```julia-repl
cs = ComputationSettings(location = "slurm", 
num_procs = 16,
num_tasks = 8,
num_threads = 8,
maxmem = 70, 
clustermanager_settings = Dict(:A => "x",
 :job_name => "y",
 :nodes => "4",
 :ntasks_per_node => "4",
 :cpus_per_task => "8",
 :exclusive => "",
 :mem => "90GB",
 :time => "23:59:59",
 :partition => "z"))
```
We have specified the following options:
- `location = "slurm"` the computation should be run on Slurm manager
- `num_procs = 16` the total number of processes to be started (16 in this case). On Slurm this has to be equal to `:nodes * :ntasks_per_node`
- `num_tasks = 8` the number of tasks per process
- `num_threads = 8` number of threads to be started in each Julia process. On Slurm this has to be equal to `:cpus_per_task` (see below)
- `maxmem = 70` specifies the level in GB where aggressive garbage collection is triggered, should be less than `:mem` (see below) to avoid using more than the allocated resources
- `clustermanager_settings` is a flexible `Dictionary` which passes the relevant options to Slurm. In this case we have specified:
    - `:A` the project account to be charged for the computational allocation requested
    - `:job_name` the name of the job 
    - `:nodes` how many HPC nodes are to be used (4 in this case)
    - `:ntasks_per_node` how many processes per node have to be started (4 in this case), must be less than the number of cores per node 
    - `:cpus_per_task` how many cores are to be used per process (8 in this case), must be less than number of cores on node
    - `:exclusive` that the job allocation cannot share nodes with other running jobs
    - `:mem` the total memory requested per node
    - `:time` the total time requested
    - `:partition` the name of the HPC partition to use
    See the Slurm [docs](https://slurm.schedmd.com/documentation.html) for more details and options.

Users can thus perform multiprocessing, multithreading and/or a combination of the two also on a cluster which uses Slurm by properly specifying these options. Including `ComputationSettings` defined in the way just explained in the `estimation` command will automatically ensure that the latter is run with Slurm.

!!! note 
    The word *task* has different meanings in Slurm and in this package: in our package (in line with Julia terminology) it refers to the number of tasks for multithreading, while in Slurm it means the number of processes per node.


!!! note 
    Applying this package with different cluster managing systems than Slurm should be possible via a slight modification of the `Distributed.addprocs(cs::ComputationSettings)` function. Any related pull request is greatly appreciated!

!!! note
    - Hardware configuration and rules for Slurm options to be included might differ across HPCs. The user should make sure that the options conform with their specific case. 
    - If on a cluster, it's important to remember to set up correctly the required environment by loading the packages and functions before running the estimation.
    - If one runs the code from an open Julia session in the HPC then the Slurm command called is `srun`. It should be possible to use also `sbatch` by writing a script that calls the code.

!!! tip
    The best combination of options to choose in `ComputationSettings` (both when running jobs locally and on HPC) depends on the specific model and computer configuration used. For instance, while setting up multiple processes enhances parallelization, initializing them also requires time. We encourage users to experiment different combinations to figure out which one is the best for their setting.

## Other useful features

### [Estimating alternative specifications](@id Example.Alternative)
The package allows easy estimation of alternative model specifications or using a different set of moments. For instance, imagine that we want to estimate the original model without the noise, i.e., $\sigma_\nu=0$, by targeting only the variance and the first-order autocovariance. The first step is to define appropriately the structure `EstimationSetup`:
```@example
setup_noise_off = EstimationSetup(AR1Estimation("ar1estim"), "noise_off", "onlytwo");
nothing # hide
```
The first element is `EstimationMode` (as we had before, and we keep it the same since we are considering a restricted version of the original model), the second element is a string specifying how we want to call this restricted specification (`modelname`), and the latter is a string specifying how to call the set of moments to target (`typemom`). Note that these two strings were defined as empty in the case presented before.

!!! note
    The `modelname` and `typemom` strings will be automatically included in the names of saved files.

Then, one specifies which parameters have to be estimated (relative to the order specified in `parambounds`) with the function `indexvector`:
```@example
function MomentMatching.indexvector(mode::AR1Estimation, modelname::String)
    indexvec = fill(true, length(parambounds(mode)[1]))
    
    if modelname == "noise_off" # we do not estimate the noise variance (which is the third element in parambounds)
        indexvec[3] = false
    end

    return indexvec
end
```
!!! note 
    If `indexvector` is not specified by the user, the algorithm includes all the parameters in `parambounds`. 

Similarly, the set of moments to be targeted can be specified by redefining `MomentMatching.datamoments` for different `typemom`:
```@example
function MomentMatching.datamoments(mode::AR1Estimation, typemom::String)
    momtrue = [0.8, 0.45, 0.4] # made up numbers

    mmomtrue = deepcopy(momtrue)

    if typemom == ""
        return hcat(momtrue, mmomtrue)
    elseif typemom == "onlytwo" # in this case only first two moments used
        return hcat(momtrue[1:2], mmomtrue[1:2])
    end
end
```
In this case we target only the first two moments. Finally, we redefine also the objective function:
```@example
function MomentMatching.obj_mom!(mom::AbstractVector, momnorm::AbstractVector,
 mode::AR1Estimation, x::Array{Float64,1}, modelname::String, typemom::String,
  aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers;
   saving_model::Bool=false, filename::String="")
    
    if modelname == ""
        (ρ, σϵ, σν) = x
    elseif modelname == "noise_off" # we set the last parameter to zero
        (ρ, σϵ, σν) = vcat(x, 0.0)
    end

    for n in 1:aux.Nsim
        preal.z[n] = 0.0
    end
    for t in 1:aux.Tsim
        for n in 1:aux.Nsim
            preal.z[n] = ρ * preal.z[n] + σϵ * presh.ϵs[n, t]
            preal.y[n] = preal.z[n] + σν * presh.νs[n, t]
        end
        if t > 2
            preal.mat[3, t] = cov(preal.y, preal.ylag2)
            copy!(preal.ylag2, preal.ylag1)
        end
        if t > 1
            preal.mat[2, t] = cov(preal.y, preal.ylag1)
            copy!(preal.ylag1, preal.y)
        end
        preal.mat[1, t] = var(preal.y)
        copy!(preal.ylag1, preal.y)
    end

    mom[1] = mean(@view preal.mat[1, aux.Tdis:end])
    momnorm[1] = mom[1]

    mom[2] = mean(@view preal.mat[2, aux.Tdis:end])
    momnorm[2] = mom[2]

    if typemom == "" # this moment is needed only in the benchmark case
        mom[3] = mean(@view preal.mat[3, aux.Tdis:end])
        momnorm[3] = mom[3]
    end

end

```
!!! note
    Besides the number of estimated parameters, `modelname` could also influence in which way the moments determined by `typemom` are computed, making more general robustness checks also easy to implement.

We also redefine the function to present the results: 
```@example
function MomentMatching.momentnames(mode::AR1Estimation, typemom::String)
    moments = fill("Cov(y_t,y_t-j)", 3)
    lags = string.(0:2)
    if typemom == ""
        return DataFrame(Moment=moments, Lags=lags)
    elseif typemom == "onlytwo"
        return DataFrame(Moment=moments[1:2], Lags=lags[1:2])
    end
end
```
!!! note 
    In a similar fashion, it is possible to make `modelname`- and `typemom`-specific also the auxiliary functions and the default matrix.

We are now ready to run the estimation of the restricted model: 
```@example
est_noise_off = estimation(setup_noise_off; npmm=npest, saving=false);
tableest(setup_noise_off, est_noise_off)
```
```@example
tablemoms(setup_noise_off, est_noise_off)
```

Alternatively, one could also estimate the restricted model by targeting the original 
three moments as follows. In this case the system is overidentified, and hence the moments cannot be matched exactly.

```@example
setup_noise_off_threemoments = EstimationSetup(AR1Estimation("ar1estim"), "noise_off", "");
est_noise_off_threemoments = estimation(setup_noise_off_threemoments; npmm=npest, saving=false);
tableest(setup_noise_off_threemoments, est_noise_off_threemoments)
```
```@example
tablemoms(setup_noise_off_threemoments, est_noise_off_threemoments)
```

### [Only global or only local](@id Example.Onlyone)
In the main example above both the global and local stages were performed in the same call. It is possible to perform only the global or only the local stage with the options `onlyglo` and `onlyloc` available in `NumParMM`: 
```@example
npest_glo = NumParMM(setup; Nglo=100, onlyglo=true)
npest_loc = NumParMM(setup; onlyloc=true,local_opt_settings = (algorithm = NelderMead(), maxtime = 30.0))

est_glo = estimation(setup; npmm=npest_glo, saving=false)
# use the best 10 global as starting points
est_loc = estimation(setup; npmm=npest_loc, xlocstart = est_glo.xglo[1:10], saving=false) 
nothing # hide
```
Note that in this example results might differ slightly from the estimation above because new shocks have been drawn (and because of the low maximum time - for exemplificatory purposes - specified for the solver in the local stage). It is possible to draw the shocks once and then pass them across different calls of `estimation` with the `presh` option.

### Merging results
For very long estimation exercises it can be useful to split the evaluation of global and/or local points across different calls of `estimation` and save the results after each call (so that if something goes wrong one does not need to recompute everything from scratch). For instance, to evaluate 10000 global points one can call `estimation` four times, each time evaluating 2500 points and then saving the results in one merged file (choosing which global points to evaluate in a given parameter space can be achieved through the option `sobolinds` in `estimation`). The function to achieve this is `mergeglo`. Below an example with 100 global points evaluated with two calls:

```@example
npest_glo_batch1 = NumParMM(setup; sobolinds=1:50, onlyglo=true)
npest_glo_batch2 = NumParMM(setup; sobolinds=51:100, onlyglo=true)

est_batch1 = estimation(setup; npmm=npest_glo_batch1, saving=false) 
est_batch2 = estimation(setup; npmm=npest_glo_batch2, saving=false)

estmerged = mergeglo(setup, [est_batch1, est_batch2]; saving=false)
nothing # hide
```
In this case, the estimation results to be merged were already in memory when merging, but one can of course load any already saved estimation result (again, note that results might be different from previous estimations for the same reasons described before).

A similar procedure can be applied for the local stage with the function `mergeloc` (in this case the user needs to specify the starting points to be evaluated with the option `xlocstart` in `estimation`). Finally, the function `mergegloloc` allows to merge together separate global and local results.

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

It is crucial that the same set of shocks are used during the parameter estimation, as otherwise convergence cannot be achieved in the local minimization phase. (The sensitivity of results to different draws of shocks can be checked via bootstrapping, as explained later in the [`Inference`](@ref Example.Inference) section.) This is again done by defining an appropriate subtype of an existing abstract type and a function generating a default container of shocks. In this case, one needs to draw a normal shock for ``\varepsilon`` and ``\nu`` for each `t` and `n`.

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

In order to compute the necessary moments of large samples, one often needs to populate large arrays with realized values(in our case, of ``y_{i,t}``s). Creating separate containers for each guess for the parameter vector would be very costly, so instead this is done once before starting the estimation, and the data contained within will be repeatedly overwritten. (Note that when performing an estimation via parallel computing, these containers are internally generated separately for each thread, and hence data race is avoided.) In this example, we will compute cross-sectional moments in each time period and take their time-average in the final step. Therefore, we need to keep track of ``z`` and ``y`` (together its first and second lags) and the already computed moments. Defining the structure of preallocated data follows a similar logic as the previous steps.

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

!!!note
    By default the deviation between data and model moments is obtained by rescaling the difference between the two with data means of each moment. For instance, if one targets the time-series of the cross-sectional skewness of the distribution of income growth, the differences in each year would be scaled by the time-series average of the cross-sectional skewness (clearly, if one targets just a moment the mean is the moment itself). The user can change this by writing a mode-specific `mdiff` function (see related code in `estimation.jl`).

## [Estimation](@id Example.Estimation)

After defining an estimation setup and a structure supplying numerical settings, one can perform the estimation as follows. After checking 100 points in the global phase, a local minimization takes place using the Nelder-Mead algorithm, stated from the 10 global points with the lowest objective function values. 

!!!note 
    In this example we use the default weighting matrix - which is the unitary matrix - but the user can change this by defining a a mode-specific `default_weight_matrix` function (see related code in `estimation.jl`) or by passing their preferred weighting matrix via the keyword argument `Wmat` in `estimation`.

```@example
using OptimizationOptimJL
setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npest = NumParMM(setup; Nglo=100, Nloc=10,
 local_opt_settings = (algorithm = NelderMead(), maxtime = 30.0))

est = estimation(setup; npmm=npest, saving=false); nothing # hide
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

Results can be saved by setting `saving` equal to `true`. In this case `filename` specified in the estimation mode will be used as suffix. The default saving paths is `"./saved/estimation_results/"`. The name will also include the model name and the name of the set of moments matched (see section [`Estimating alternative specifications`](@ref Example.Alternative)).

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
boot = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false)

fbootstrap(setup, est, boot)
savefig("fbootstrap.svg"); nothing # hide
```
![](fbootstrap.svg)

## [Multithreading and multiprocessing](@id Example.Multi)
The global and local phases of the estimation procedure require evaluating the objective function at many points of the parameter space. In our package this task can be parallelized with multithreading (`Threads` module, distributes across cores within a process), multiprocessing (`Distributed` module, distributes across different processes) and a combination of the two (distributes across different processes and then across the cores within a process). We describe below how to implement each locally and on a cluster. 

!!! danger 
    While we have designed the package to make it hard to create data races, it is always the user's responsibility to check that this does not happen in their own model. For instance, it is not suggested to solve a model with multithreading if the latter is already active when looping over points in the parameter space. Use the macro `maythread` in the part of your code using multithreading to be able to set the latter on and off with the function `threading_inside` (both are included in the package).

### Local parallelization
The following code performs the global stage locally (`location="local"`) in three ways:
1. Two processes (`num_procs=2`), each started with two threads (i.e., cores `num_threads=2`) and no multithreading (`num_tasks=1`).
2. One process (default) with all avaliable threads (default with one process) and multithreading (`num_tasks` defaults to twice the number of threads).
3. Two processes, each started with two threads and multithreading.
!!! note
    Multiprocessing distributes the tasks equally across the number of processes.
!!! note
    The default option for `num_tasks` tries to minimize idleness and implies that multithreading is active by default.

The difference between the first and the third case is that in the latter evaluation of points is distributed across the two threads while in the former two cores are used but evaluation of points is not parallelized across them.

Given that memory is not shared across the different processes, before running any code using multiprocessing we need to make sure that the required elements (functions, packages, structures, types...) are loaded in each of them. The function to do that is `load_on_procs`. Specifically, one writes a Julia script dedicated to loading all the required elements and calls it in `load_on_procs` which takes care of running it in every process. In our case such file is called `init.jl` and it basically loads the functions, packages, structures, types, etc. that we have used so far in this example (the script is available in the `docs/src` folder of the GitHub repository of the package).


```@example
using Distributed
function MomentMatching.load_on_procs(mode::AR1Estimation)
    return @everywhere begin include("./docs/src/init.jl") end
end
```

Now we are ready to perform the estimation in all the three ways just described.

```@example
cs_1 = ComputationSettings(location="local", num_procs=2, num_threads=2,num_tasks=1)
cs_2 = ComputationSettings(location="local")
cs_3 = ComputationSettings(location="local", num_procs=2, num_threads=2)

# common shocks to test equivalence of methods
auxest = AuxiliaryParameters(AR1Estimation("ar1estim"), "")
preshest = PredrawnShocks(AR1Estimation("ar1estim"), "", "", auxest)

est_1 = estimation(setup; npmm=npest, presh=preshest, cs=cs_1, saving=false)
est_2 = estimation(setup; npmm=npest, presh=preshest, cs=cs_2, saving=false)
est_3 = estimation(setup; npmm=npest, presh=preshest, cs=cs_3, saving=false)
```

Note that Julia informs the user with a message whenever a process is started. The results are of course identical across the different specifications (they might differ slightly from the estimation above since we have drawn new shocks). 

Since in the code above both the global and local phases are performed, the specified computational settings are applied to both. It is of course possible to run each phase separately with its own computational settings. `ComputationSettings` also works in the function performing bootstrapping.

!!! tip
    Choosing the best combination of number of processes, threads and tasks depends on the specific model and computer configuration used. For instance, while setting up multiple processes enhances parallelization, initializing them also requires time. We encourage users to experiment different combinations to figure out which one is the best for their setting.

### Parallelization on a cluster
Currently, our package works only on clusters using Slurm Workload Manager. 

## Ease of use

### Only global or only local
In the main example above both the global and local stages were performed in the same call. It is possible to perform only the global or only the local stage with the options `onlyglo` and `onlyloc` available in `NumParMM`: 
```@example
setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npest_glo = NumParMM(setup; Nglo=100, onlyglo=true)
npest_loc = NumParMM(setup; onlyloc=true,local_opt_settings = (algorithm = NelderMead(), maxtime = 30.0))

est_glo = estimation(setup; npmm=npest_glo, saving=false)
# use the best 10 global as starting points
est_loc = estimation(setup; npmm=npest_loc, xlocstart = est_glo.xglo[1:10], saving=false); nothing # hide
```
Note that in this example results might differ slightly from the estimation above because new shocks have been drawn. It is possible to draw the shocks once and then pass them across different calls of `estimation` with the `presh` option. See the section [`Multithreading and multiprocessing`](@ref Example.Multi) for an example.  

### Merging results
For very long estimation exercises it can be useful to split the evaluation of global and/or local points across different calls of `estimation` and save the results after each call (so that if something goes wrong one does not need to recompute everything from scratch). For instance, to evaluate 10000 global points one can call `estimation` four times, each time evaluating 2500 points and then saving the results (choosing which global points to evaluate in a given parameter space can be achieved through the option `sobolinds` in `estimation`). The function to achieve this is `mergeglo`. Below an example with 100 global points evaluated with two calls:

```@example
npest_glo_batch1 = NumParMM(setup; sobolinds=1:50, onlyglo=true)
npest_glo_batch2 = NumParMM(setup; sobolinds=51:100, onlyglo=true)

est_batch1 = estimation(setup; npmm=npest_glo_batch1, saving=false) 
est_batch2 = estimation(setup; npmm=npest_glo_batch2, saving=false)

estmerged = mergeglo(setup, [est_batch1, est_batch2]; saving=false); nothing # hide
```
In this case, the estimation results to be merged were already in memory when merging, but one can of course load any already saved estimation result (again, note that results might be different from previous estimations because new shocks were drawn).

A similar procedure can be applied for the local stage with the function `mergeloc` (in this case the user needs to specify the starting point with the option `xlocstart` in `estimation`). Finally, the function `mergegloloc` allows to merge together separate global and local results.

!!! danger
    To create the merged estimation result structure, `mergeglo` uses AuxiliaryParameters, PredrawnShocks. and ParMM (the latter is an auxiliary structure to initialize all the estimation inputs, see relevant code in `estimation.jl`) structures from first entry of the vector of results. The same holds for `mergeloc` with the addition that also the global results used as input come from the first entry of the vector of results. Finally `mergegloloc` assumes that global and local results to be merged are already ordered (`mergeglo` and `mergeloc` instead obviously perform reordering).

### [Estimating alternative specifications](@id Example.Alternative)

### Wrapping any model

## Relation with GMM
In the above example we have explained how the procedure works with SMM, but extending usage of our package routines to GMM is straightforward. With GMM usually one has a set of moment conditions that should hold with equality and rather than simulating data from a model, actual data are used to compute such conditions over the points in the parameter space. 

The user, therefore, in this case just needs to write their own code to compute such conditions and check how far away from zero they are. In other words, zero is the data moment to be used when computing the difference between model and data moments. 

Note that since the default version of `mdiff` scales by the average of a specific data moment (if one targets just a moment the mean is clearly the moment itself), if such value is zero, the user also needs to write a mode-specific `mdiff` function.


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

During the estimation, one needs to evaluate the objective function for each parameter guesses. Passing invariant parameters to the objective function is possible via defining an auxiliary structure which has to be a subtype of `AuxiliaryParameters`. One also needs to write a corresponding function to generate a default auxiliary structure as shown below. In this case, we pass the dimensions of the simulated sample.

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

It is crucial that the same set of shocks are used during the parameter estimation, as otherwise convergence cannot be achieved in the local minimization phase. (The sensitivity of results to different draws of shocks can be checked via bootstrapping, as explained later in the [`Inference`](@ref) section.) This is again done by defining an appropriate subtype of an existing abstract type and a function generating a default container of shocks. In this case, need to draw a normal shock for ``\varepsilon`` and ``\nu`` for each `t` and `n`.

```@example
struct AR1PreShocks{S<:AbstractFloat} <: PredrawnShocks
    "preallocated array for persistent shocks"
    ϵs::Array{S,2}
    "preallocated array for transitory shocks"
    νs::Array{S,2}
end

function PredrawnShocks(mode::AR1Estimation, modelname::String, typemom::String, aux::AuxiliaryParameters)
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

function PreallocatedContainers(mode::AR1Estimation, modelname::String, typemom::String, aux::AuxiliaryParameters)

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
function MomentMatching.obj_mom!(mom::AbstractVector, momnorm::AbstractVector, mode::AR1Estimation, x::Array{Float64,1}, modelname::String, typemom::String, aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers; saving_model::Bool=false, filename::String="")
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

We give the names and ranges of the targeted parameters by writing a method of `parambounds`. During the global phase of the estimation, the region within 'global' bounds is searched. Violating 'hard' bounds during the local phase induces a penalty to redirect the algorithm towards the allowed range during the local search.

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
    momtrue = [0.8, 0.6, 0.4] # made up numbers

    mmomtrue = deepcopy(momtrue)

    return hcat(momtrue, mmomtrue)
end
```

Finally, we name the targeted moments as below. The `momentnames` function has to return a `DataFrame` with two columns, where one targeted moment corresponds to one row. If the two moments have coinciding values in the first column, the corresponding results will be visualized together, as shown in section [`Estimation`](@ref Example.Estimation).

```@example
using DataFrames
function MomentMatching.momentnames(mode::AR1Estimation, typemom::String)
    moments = fill("Cov(y_t,y_t-j)", 3)
    lags = string.(0:2)
    return DataFrame(Moment=moments, Lags=lags)
end
```


## [Estimation](@id Example.Estimation)

After defining an estimation setup and a structure supplying numerical settings, one can perform the estimation as follows. After checking 100 points in the global phase, a local minimization takes place using the Nelder-Mead algorithm, stated from the 10 global points with the lowest objective function values.

```@example
using OptimizationOptimJL
setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npest = NumParMM(setup; Nglo=100, Nloc=10, local_opt_settings = (algorithm = NelderMead(), maxtime = 30.0))

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

## Diagnostics

When the objective function is highly non-linear, it is in general difficult to know if the obtained parameter estimate indeed corresponds to a global minimizer. One concern would be that the obtained local optimum is 'too local', i.e. its basin of attraction is too narrow. In this case the local optima would be very sensitive. To judge the accuracy of the estimated parameter vector, two heuristic methods are available.

First, it is possible to visualize how the objective function depends on varying the parameter estimates one-at-a-time (keeping the other parameters constant), around the best point.

```@example
marg = marginal_fobj(setup, est, 17, fill(0.1, 3))
fmarg(setup, est, marg)
savefig("fmarg.svg"); nothing # hide
```
![](fmarg.svg)

Second, one can visualize how sensitive the corresponding parameter values are to the rank of the corresponding global or local point. This is informative on the sufficient number of global and local points.

```@example
fsanity(setup, est, glob = true)
savefig("fsanity_glo.svg"); nothing # hide
```
![](fsanity_glo.svg)

```@example
fsanity(setup, est)
savefig("fsanity_loc.svg"); nothing # hide
```
![](fsanity_loc.svg)

## Inference

### Parametric Bootstrap

```@example
Tdis = 20 # burn in
Tdata = 40 # true data length
Ndata = 500 # true sample size
Nsample = 5 # number of samples used for bootstrap
Nseed = 5 # number of shock simulations used for bootstrap
auxmomsim = AR1AuxPar(Ndata, Tdata + Tdis, Tdis)
boot = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false)

fbootstrap(setup, est, boot)
savefig("fbootstrap.svg"); nothing # hide
```
![](fbootstrap.svg)

### Asymptotic Inference
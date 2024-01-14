#= This file performs parameter inference and computes other diagnostics on the estimation.
=#

# FUNCTIONS TO COMPUTE OBJECTIVE AT POINTS AROUND THE MINIMIZER

"""
$(TYPEDSIGNATURES)

Computes objective function at points around the minimizer.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
    
# Optional arguments
- gridx: Grid of points around of optimum to evaluate objective function at.
"""
function marginal_fobj(estset::EstimationSetup, mmsolu::EstimationResult; gridx = defaultxgrid(mmsolu.xloc[1]))
    @unpack floc, xloc, momloc, pmm, aux, presh = mmsolu
    ob = floc[1]
    x = xloc[1]

    preal = PreallocatedContainers(estset,aux)
    if typeof(presh) == EmptyPredrawnShocks
        presh = PredrawnShocks(estset, aux)
        @warn("marginal plot is less precise when built from lightweight result")
    end
    
    objvals = Array{Float64}(undef, size(gridx,1), length(x))
    
    for j in eachindex(x)
            
        for i in axes(gridx,1)

            xfori = deepcopy(x)
            xfori[j] = gridx[i,j]

            if i == div(size(gridx,1)+1,2,RoundNearest) # for point corresponding to argmin we use the optimal value
                objvals[i, j] = ob
            else
                objvals[i, j] = objf!(fill(0.0,length(momloc[1])),fill(0.0,length(momloc[1])),estset, xfori, pmm, aux, presh, preal)
            end

        end
    end

    return [gridx;;;objvals]
end

"""
$(TYPEDSIGNATURES)

Computes objective function at points around the minimizer.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- num_marg: Number of points around optimum to evaluate.
- scale_margs: Scale parameters for constructing grid of points to be evaluated.
"""
function marginal_fobj(estset::EstimationSetup, mmsolu::EstimationResult, num_marg::Integer, scale_margs::AbstractVector)
    return marginal_fobj(estset, mmsolu; gridx = defaultxgrid(mmsolu.xloc[1], num_marg = num_marg, scale_margs = scale_margs))
end

"""
$(TYPEDSIGNATURES)

Computes objective function at points around the minimizer.

# Required arguments
- x: Array with parameter estimates.
- num_marg: Number of points around optimum to evaluate.
- scale_margs: Scale parameters for constructing grid of points to be evaluated.
"""
function defaultxgrid(x::Vector{Float64}; num_marg::Integer=9, scale_margs::AbstractVector = fill(0.1,length(x)))
    iseven(num_marg) && throw(error("Need odd number of grid points!"))

    xvals = Array{Float64}(undef, num_marg, length(x))

    for j in eachindex(x)
        skip = abs(x[j]) * scale_margs[j]
        gridx = collect(range(x[j] - skip, x[j] + skip, num_marg))
        xvals[:,j] = gridx
    end
    return xvals
end

# FUNCTIONS FOR BOOTSTRAP, STANDARD ERRORS AND TESTS

"""
$(TYPEDSIGNATURES)

Performs parametric bootstrap for inference purposes. Outputs a tuple including:
- an array containing parameter guesses of dimensions xdim * Nseeds * Nsamplesim. Can be used to compute bootstrap confidence intervals.
- an array containing moments from simulations of dimensions momdim * Nsamplesim. Can be used to compute the efficient weighting matrix and asymptotic confidence intervals.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- auxmomsim: Contains the numerical parameters used to simulate the distribution of moments when model and the estimated parameters are assumed to be correct. Its format is not restricted, but `defaultInnerAuxStructure(mode,auxmomsim)` should return a valid aux structure. Should imitate the process generating the data, based on which the original moments were computed. 
- nprepeat: Contains the numerical parameters used in simulation when reestimating the parameters of the model with alternative moments to match and different seeds. Its format is not restricted, but `defaultInnerAuxStructure(mode,auxmomsim)` should return a valid aux structure. Should imitate the setup of the original estimation.
- Nseeds: Number of different seeds to try for each alternative sample moment vector.
- Nsamplesim: Number of alternative sample moment vectors to try.

# Explanation. Two phases: 
- 1. We are interested in how precisely the targeted moments were estimated in the original data. 
Therefore we assume model and estimated parameters are correct and simulate alternative samples, 
same size as original sample from data. For each, we compute the alternative moment vector to match.
- 2. For each alternative moment vector, we repeat the estimation procedure (or at least the local phase of it). 
The resulting distribution of estimated parameters gives us the bootstrap confidence intervals. 
For diagnostic reasons, we repeat this procedure with Nseeds number of seeds, 
if results vary a lot with different seeds, it is recommended to restart the whole estimation with bigger simulation parameters.
"""
function param_bootstrap(estset::EstimationSetup, mmsolu::EstimationResult,auxmomsim::AuxiliaryParameters, Nseeds::Integer, Nsamplesim::Integer)
    @unpack mode, modelname, typemom = estset
    @unpack aux, xloc, npmm = mmsolu
    bestx = xloc[1]
    momleng = length(mmsolu.pmm.momdat)

    moms = [Vector{Float64}(undef,momleng) for sample_i in 1:Nsamplesim]
    momnorms = [Vector{Float64}(undef,momleng) for sample_i in 1:Nsamplesim]
    prealc = PreallocatedContainers(estset,auxmomsim)
    for sample_i in 1:Nsamplesim
        obj_mom!(moms[sample_i],momnorms[sample_i],mode, bestx, modelname, typemom, auxmomsim, PredrawnShocks(estset,auxmomsim), prealc)
    end

    mms = [initMMmodel(estset, npmm, moms2 = hcat(moms[sample_i],momnorms[sample_i])) for sample_i in 1:Nsamplesim] # re-estimation initiated with alternative moments instead of moments from data

    presh_repeat = [PredrawnShocks(estset,aux) for seed_i in 1:Nseeds] # Nseeds different aux structure for each alternative moment

    prog = Progress(Nsamplesim*Nseeds; desc="Bootstrapping...", color=:blue)

    xs = Array{Float64}(undef,length(mmsolu.xloc[1]),Nseeds,Nsamplesim)
    chunks = getchunks(Nsamplesim*Nseeds)

    @assert(!threading_inside())

    tasks = map(chunks) do chunk
        Threads.@spawn begin
            x_ch = [Vector{Float64}(undef,length(mmsolu.xloc[1])) for _ in 1:length(chunk)]
            preal = PreallocatedContainers(estset,aux)
            for n in eachindex(chunk)
                fullind = chunk[n]
                seed_i = CartesianIndices((Nseeds,Nsamplesim))[fullind][1]
                sample_i = CartesianIndices((Nseeds,Nsamplesim))[fullind][2]
                x = [[1.0]]

                opt_loc!([1.0], x, [Vector{Float64}(undef,momleng)], [Vector{Float64}(undef,momleng)], [false], estset, npmm.it, aux, presh_repeat[seed_i], preal, mms[sample_i], bestx, 1)

                x_ch[n] = x[1]
                ProgressMeter.next!(prog)
            end
            return x_ch
        end
    end
    outstates = fetch.(tasks)
    finish!(prog)

    for (i,chunk) in enumerate(chunks)
        for n in eachindex(chunk)
            fullind = chunk[n]
            seed_i = CartesianIndices((Nseeds,Nsamplesim))[fullind][1]
            sample_i = CartesianIndices((Nseeds,Nsamplesim))[fullind][2]
            xs[:, seed_i, sample_i] = outstates[i][n]
        end
    end

    return xs, hcat(moms...)
end

"""
$(TYPEDSIGNATURES)

Compute the covariance matrix of the moments.

# Required arguments
- moms: Array with simulated moments.
"""
function Omega_boots(moms::AbstractArray)
    cv = cov(moms; dims=2)
    return cv
end

"""
$(TYPEDSIGNATURES)

Compute the efficient covariance matrix.

# Required arguments
- moms: Array with simulated moments.
"""
function efficient_Wmat(moms::AbstractArray)
    return inv(Omega_boots(moms))
end

"""
$(TYPEDSIGNATURES)

Matrix containing the derivatives of the objective function with respect to the parameter vector.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
"""
function Qmatrix(estset::EstimationSetup, mmsolu::EstimationResult)
    @unpack mode, modelname, typemom = estset
    @unpack aux, presh, xloc, pmm = mmsolu
    bestx = xloc[1]
    ϵ = 10^-5

    Q = Array{Float64}(undef,length(pmm.momdat),length(bestx))

    for xi in axes(Q,2)
        xperturb = zeros(length(bestx))
        xperturb[xi] = ϵ

        upmoms = obj_mom(mode, bestx.+xperturb, modelname, typemom, aux, presh)
        downmoms = obj_mom(mode, bestx.-xperturb, modelname, typemom, aux, presh)

        for mi in axes(Q,1)
            Q[mi,xi] = (upmoms[mi]-downmoms[mi])/(2*ϵ)
        end
    end

    return Q
end

"""
$(TYPEDSIGNATURES)

Compute the sandwich estimator of the covariance matrix.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- moms: Array with moments. (DEFINE TYPE IN THE FUNCTION)
"""
function sandwich_matrix(estset::EstimationSetup, mmsolu::EstimationResult, moms)
    Q = Qmatrix(estset, mmsolu)
    W = mmsolu.pmm.W
    Ω = Omega_boots(moms)
    bread0 = Q'*W*Q
    cond(bread0) > 10^5 && @warn("Standard errors might be imprecise due to ill-conditioned bread matrix. Condition number is $(cond(bread0))")
    bread = inv(bread0)
    meat = Q'*W*Ω*W*Q
    return bread*meat*bread
end

"""
$(TYPEDEF)
# Description
Structure to store bootstrap results.
# Fields
$(FIELDS)
"""
@with_kw struct BootstrapResult{S<:Real}
    "simulated moments"
    moms::Array{S,2}
    "simulated parameter values"
    xs::Array{S,3}
    "asymptotic standard errors"
    sd_asymp::Vector{S}
    "efficient weighting matrix"
    W::Array{S,2}
end

"""
$(TYPEDSIGNATURES)

Performs bootstrap and computes related quantities.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- auxmomsim: Contains the numerical parameters used to simulate the distribution of moments when model and the estimated parameters are assumed to be correct. Its format is not restricted, but `defaultInnerAuxStructure(mode,auxmomsim)` should return a valid aux structure. Should imitate the process generating the data, based on which the original moments were computed. 
- nprepeat: Contains the numerical parameters used in simulation when reestimating the parameters of the model with alternative moments to match and different seeds. Its format is not restricted, but `defaultInnerAuxStructure(mode,auxmomsim)` should return a valid aux structure. Should imitate the setup of the original estimation.
- Nseeds: Number of different seeds to try for each alternative sample moment vector.
- Nsamplesim: Number of alternative sample moment vectors to try.
- Ndata: Size of data sample.

# Optional arguments
- saving: Logical, true if results are to be saved.
- filename_suffix: String with suffix to be used for file name when saving.
"""
function param_bootstrap_result(estset::EstimationSetup, mmsolu::EstimationResult,auxmomsim::AuxiliaryParameters, Nseeds::Integer, Nsamplesim::Integer, Ndata::Integer; saving::Bool = false, filename_suffix::String="")
    @unpack mode, modelname, typemom = estset
    @unpack npmm = mmsolu
    xs, moms = param_bootstrap(estset, mmsolu,auxmomsim, Nseeds, Nsamplesim)
    sm = sandwich_matrix(estset, mmsolu,moms)
    W = efficient_Wmat(moms)
    bootres = BootstrapResult(moms,xs,sqrt.(diag(sm)/Ndata),W)
    saving && save(estimation_result_path() * estimation_name(estset,npmm, filename_suffix) * "_bootstrap" *  ".jld", "bootres", bootres)
    return bootres
end

"""
$(TYPEDSIGNATURES)

Perform Hansen-Sargan test for overidentifying restrictions.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- boot: Instance of BootstrapResult. See separate documentation [`BootstrapResult`](@ref).
- n: Rescaling parameter for J-statistic (data size).
"""
function Jtest(estset::EstimationSetup, mmsolu::EstimationResult, boot::BootstrapResult,n::Integer)
    @unpack mode, modelname, typemom = estset
    @unpack pmm, xloc, momloc = mmsolu
    @unpack W = boot
    g = mdiff(mode, momloc[1], pmm.momdat, pmm.mmomdat)
    J = n*g'*W*g
    df = length(momloc[1])-length(xloc[1])
    df == 0  && throw(DimensionMismatch("J-test statistic cannot be computed for exactly identified models."))
    p = ccdf(Chi(df), J)
    tab = DataFrame(Symbol("J statistic") => J,Symbol("Degrees of freedom") => df, Symbol("p-value") => p)
    display(tab)
    return tab
end
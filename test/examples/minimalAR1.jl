using Statistics, DataFrames
using MomentMatching, Distributed # so that include line uses these packages in load_on_procs

struct AR1Estimation <: EstimationMode
    "mode-dependent prefix of filenames used for saving estimation results"
    filename::String
end

function MomentMatching.load_on_procs(mode::AR1Estimation)
    return @everywhere begin include("examples/minimalAR1.jl") end
end

struct AR1AuxPar{T<:Integer} <: AuxiliaryParameters
    "number of agents to simulate"
    Nsim::T
    "number of time periods to simulate"
    Tsim::T
    "number of periods to discard for moment evaluation "
    Tdis::T
end

MomentMatching.AuxiliaryParameters(mode::AR1Estimation, modelname::String) = AR1AuxPar(10000, 200, 100)

struct AR1PreShocks{S<:AbstractFloat} <: PredrawnShocks
    "preallocated array for persistent shocks"
    ϵs::Array{S,2}
    "preallocated array for transitory shocks"
    νs::Array{S,2}
end

function MomentMatching.PredrawnShocks(mode::AR1Estimation, modelname::String, typemom::String, aux::AuxiliaryParameters)
    return AR1PreShocks(randn(aux.Nsim, aux.Tsim),
        randn(aux.Nsim, aux.Tsim))
end

function MomentMatching.parambounds(mode::AR1Estimation)
    full_labels    = [ "ρ",  "σϵ",  "σν"]
    full_lb_hard   = [ 0.0,   0.0,  0.0 ]
    full_lb_global = [ 0.0,   0.0,  0.0 ]
    full_ub_global = [ 1.0,   1.0,  1.0 ]
    full_ub_hard   = [ 1.0,   1.0,  1.0 ]
    return full_labels, full_lb_hard, full_lb_global, full_ub_global, full_ub_hard
end

# Recover moments from the data

function MomentMatching.datamoments(mode::AR1Estimation, typemom::String)
    momtrue = [0.8, 0.45, 0.4] # made up numbers

    mmomtrue = deepcopy(momtrue)

    return hcat(momtrue, mmomtrue)
end

struct AR1PrealCont{S<:AbstractFloat} <: PreallocatedContainers
    z::Vector{S}
    y::Vector{S}
    ylag1::Vector{S}
    ylag2::Vector{S}
    mat::Array{S,2}
end

function MomentMatching.PreallocatedContainers(mode::AR1Estimation, modelname::String, typemom::String, aux::AuxiliaryParameters)

    z = Vector{Float64}(undef, aux.Nsim)
    y = Vector{Float64}(undef, aux.Nsim)
    ylag1 = Vector{Float64}(undef, aux.Nsim)
    ylag2 = Vector{Float64}(undef, aux.Nsim)

    mat = Array{Float64}(undef, 3, aux.Tsim)

    return AR1PrealCont(z, y, ylag1, ylag2, mat)
end

function MomentMatching.obj_mom!(mom::AbstractVector, momnorm::AbstractVector, mode::AR1Estimation, x::Array{Float64,1}, modelname::String, typemom::String, aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers; saving_model::Bool=false, filename::String="")
    (ρ, σϵ, σν) = x

    for n in 1:aux.Nsim
        preal.z[n] = 0.0
    end
    for t in 1:aux.Tsim
        @maythread for n in 1:aux.Nsim
            preal.z[n] = ρ * preal.z[n] + σϵ * presh.ϵs[n, t]
            preal.y[n] = preal.z[n] + σν * presh.νs[n, t]
        end
        if t > 2
            preal.mat[3, t] = cov_noalloc(preal.y, preal.ylag2)
            copy!(preal.ylag2, preal.ylag1)
        end
        if t > 1
            preal.mat[2, t] = cov_noalloc(preal.y, preal.ylag1)
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

function cov_noalloc(x,y)
    xbar = mean(x)
    ybar = mean(y)
    cov = 0.0
    N = length(x)
    for i in eachindex(x)
        cov += (x[i]-xbar)*(y[i]-ybar)
    end
    return cov/(N-1)
end

function MomentMatching.momentnames(mode::AR1Estimation, typemom::String)
    moments = fill("Cov(y_t,y_t-j)", 3)
    lags = string.(0:2)
    return DataFrame(Moment=moments, Lags=lags)
end
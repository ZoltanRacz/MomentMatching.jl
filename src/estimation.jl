#= This file contains a stand-alone routine for estimating parameters via moment matching.

Functions used outside of this file:

- estimation
- two_stage_estimation

To use this script, one needs to 
- define a subtype of abstract types EstimationMode and InnerAuxStructure (G: is it really true that the latter is required in the current version?)
and appropriate versions of functions (TO CHECK THAT THIS LIST IS COMPLETE)
- defaultInnerAuxStructure
- parambounds
- indexvector
- datamoments
- mdiff
- obj_mom
need to be defined. See modest.jl for an example.
=#

# DEFINE STRUCTURES AND TYPES 

"""
$(TYPEDEF)
# Description
Type for the model to be estimated.
"""
abstract type EstimationMode end

"""
$(TYPEDEF)
# Description
Structure to store setup of matching moments estimation procedure.
# Fields
$(FIELDS)
"""
struct EstimationSetup{U<:EstimationMode}
    "Estimation mode."
    mode::U
    "Submethod for estimation. String encoding which parameters and how are estimated."
    modelname::String
    "Submethod for estimation. String encoding which moments are targeted."
    typemom::String
end

"""
$(TYPEDEF)
# Description
Supplies non-estimated auxuliary parameters, which are needed to compute model moments for any guess of the estimated parameters. Assumed to be non-random and constant over the estimation procedure.
"""
abstract type AuxiliaryParameters end

"""
$(TYPEDSIGNATURES)

Set up default AuxiliaryParameters object. Should be defined for all estimation modes (i.e. for any subtype of EstimationMode).
"""
function AuxiliaryParameters(mode::EstimationMode, modelname::String)
    throw(error("a separate method has to be written for $(typeof(mode))"))
end

AuxiliaryParameters(estset::EstimationSetup) = AuxiliaryParameters(estset.mode, estset.modelname)

"""
$(TYPEDEF)
# Description
Supplies pre-drawn shocks, which are needed to compute model moments for any guess of the estimated parameters. Assumed to be random, but constant over the estimation procedure (so that the minimization procedure works well). They are however re-simulated when performing bootstrap.
"""
abstract type PredrawnShocks end

struct EmptyPredrawnShocks <: PredrawnShocks end

"""
$(TYPEDSIGNATURES)

Set up default PredrawnShocks object. Should be defined for specific estimation mode only if it is used.
"""
function PredrawnShocks(mode::EstimationMode, modelname::String, typemom::String, aux::AuxiliaryParameters)
    return EmptyPredrawnShocks()
end

PredrawnShocks(estset::EstimationSetup, aux::AuxiliaryParameters) = PredrawnShocks(estset.mode, estset.modelname, estset.typemom, aux)

"""
$(TYPEDEF)
# Description
Supplies empty arrays for computations to calculate model moments as few memory allocations as possible.
"""
abstract type PreallocatedContainers end

struct EmptyPreallocatedContainers <: PreallocatedContainers end

"""
$(TYPEDSIGNATURES)

Set up default PreallocatedContainers object. Should be defined for specific estimation mode only if it is used.
"""
function PreallocatedContainers(mode::EstimationMode, modelname::String, typemom::String, aux::AuxiliaryParameters)
    return EmptyPreallocatedContainers()
end

PreallocatedContainers(estset::EstimationSetup, aux::AuxiliaryParameters) = PreallocatedContainers(estset.mode, estset.modelname, estset.typemom, aux)

"""
$(TYPEDEF)
# Description
Structure to store numerical parameters for estimation procedure.
# Fields
$(FIELDS)
"""
@with_kw struct NumParMM{S<:AbstractFloat,T<:Integer}
    "Indexes of Sobol points to use"
    sobolinds::UnitRange{T}
    "# of best points to evaluate for local stage of estimation (or to save if only global stage, see [`matchmom`](@ref))"
    Nloc::T
    "Lower bound for parameters in global stage"
    full_lb_global::Vector{S}
    "Upper bound for parameters in global stage"
    full_ub_global::Vector{S}
    "Logical, true if only global optimization is to be performed"
    onlyglo::Bool
    "Logical, true if only local optimization is to be performed"
    onlyloc::Bool
    "Settings for optimization used in local stage"
    local_opt_settings::Dict{Symbol,Any}
end

"""
$(TYPEDSIGNATURES)

Create instance of NumParMM.
"""
function NumParMM(estset::EstimationSetup; Nglo::T=10000, sobolinds::UnitRange{T}=-1:-1, Nloc::T=100, onlyglo::Bool=false, onlyloc::Bool=false, full_lb_global::Vector{Float64}=parambounds(estset.mode)[3], full_ub_global::Vector{Float64}=parambounds(estset.mode)[4], local_opt_settings = Dict(:algorithm => NelderMead(), :maxiter => 10000)) where {T<:Integer}
    typeof(local_opt_settings) <: NamedTuple && (local_opt_settings = Dict(pairs(local_opt_settings)))
    sinds = ifelse(sobolinds == -1:-1, 1:Nglo, sobolinds)
    return NumParMM(sinds, Nloc, full_lb_global, full_ub_global, onlyglo, onlyloc, local_opt_settings)
end

"""
$(TYPEDEF)
# Description
Structure to store matching moments estimation inputs.
# Fields
$(FIELDS)
"""
@with_kw struct ParMM{S<:AbstractFloat}
    "Lower bound for parameters in global stage"
    lb_global::Vector{S}
    "Upper bound for parameters in global stage"
    ub_global::Vector{S}
    "Hard lower bound for parameters - being outside brings penalty in local stage"
    lb_hard::Vector{S}
    "Hard upper bound for parameters - being outside brings penalty in local stage"
    ub_hard::Vector{S}
    "Labels of estimated parameters"
    labels::Vector{String}
    "Moments from the data to match"
    momdat::Vector{S}
    "Normalizing factor moments from the data to match"
    mmomdat::Vector{S}
    "Weighting matrix"
    W::Array{S,2}
    "Predetermined quantity to re-center moment condition"
    mdifrec::Vector{S}
end



"""
$(TYPEDEF)
# Description
Structure to store output of matching moments estimation procedure.
# Fields
$(FIELDS)
"""
struct EstimationResult{S<:AbstractFloat,T<:Integer,U<:AuxiliaryParameters,V<:PredrawnShocks}
    "Numerical parameters"
    npmm::NumParMM{S,T}
    "Auxiliary inputs"
    aux::U
    "Predrawn shocks"
    presh::V
    "Starting parameters - relevant when only local stage is performed"
    xlocstart::Vector{Vector{S}}
    "Moment estimation inputs"
    pmm::ParMM{S}
    "Objective function value in global stage, sorted in increasing order"
    fglo::Vector{S}
    "Parameter combinations checked in global stage, sorted according to objective function value"
    xglo::Vector{Vector{S}}
    "Moments from model, global stage"
    momglo::Vector{Vector{S}}
    "Objective function value in local stage, sorted in increasing order"
    floc::Vector{S}
    "Parameter combinations checked in local stage, sorted according to objective function value"
    xloc::Vector{Vector{S}}
    "Moments from model, local stage"
    momloc::Vector{Vector{S}}
    "Logical, if true convergence criterion at minimum is satisfied in the local stage"
    conv::Vector{Bool}
end

"""
$(TYPEDEF)
# Description
Computational settings
# Fields
$(FIELDS)
"""
@with_kw struct ComputationSettings{T<:Integer}
    "where computation is performed. 'local' and 'slurm' are supported presently"
    location::String = "local"    
    "Number of processes. Giving 1 avoids multiprocessing (since adding only one worker would have negative effect on performance, as master is not used in the loop). On a cluster give number of nodes (Should double check this)."
    num_procs::T = 1
    "Number of tasks per process. Giving somewhat more than the number of actual ( virtual or physical ?? ) threads is probably a good idea.  "
    num_tasks::T = Threads.nthreads()*2
    "Number of threads that each processes are started with."
    num_threads::T = num_tasks
    "Trigger intensive garbage collection at this memory usage"
    maxmem::T = -1
    "Other settings"
    clustermanager_settings::Dict{Symbol,String} = Dict(:x => "")
end

## MAIN FUNCTION

"""
$(TYPEDSIGNATURES)

Estimate model parameters given instance of [`EstimationSetup`](@ref). 

Can be customized if non-default estimation cases have to be performed. Accepts initial guess(es) when only local stage is performed.
"""
function estimation(estset::EstimationSetup; npmm::NumParMM=NumParMM(estset), cs::ComputationSettings = ComputationSettings(), aux::AuxiliaryParameters=AuxiliaryParameters(estset),
    presh::PredrawnShocks=PredrawnShocks(estset, aux), xlocstart::Vector{Vector{Float64}}=[[1.0]], saving::Bool=true, saving_bestmodel::Bool=saving, number_bestmodel::Integer=1, filename_suffix::String="", errorcatching::Bool=false, vararg...)

    @assert(!threading_inside() || cs.num_tasks==1)

    pmm = initMMmodel(estset, npmm; vararg...) # initialize inputs for estimation

    mmsolu = matchmom(estset, pmm, npmm, cs, aux, presh, xlocstart, saving_bestmodel, number_bestmodel, filename_suffix, errorcatching) # perform estimation

    saving && save_estimation(estset, npmm, mmsolu, filename_suffix) # saving

    return mmsolu
end

"""
$(TYPEDSIGNATURES)

Initializes structure to store matching moments estimation inputs. See separate documentation [`ParMM`](@ref).
"""
function initMMmodel(estset::EstimationSetup, npmm::NumParMM; moms2=datamoments(estset.mode, estset.typemom), Wmat=default_weight_matrix(estset, size(moms2, 1)), mdifr=zeros(size(moms2, 1)))
    @unpack full_lb_global, full_ub_global = npmm
    @unpack mode, modelname = estset

    full_labels, full_lb_hard, full_ub_hard = parambounds(mode)[[1, 2, 5]]

    indexvec = indexvector(mode, modelname) # indexvec is a logical vector, it selects those parameters that are estimated

    mom = moms2[:, 1] # moment from data to match
    mmom = moms2[:, 2]

    return ParMM(lb_global=full_lb_global[indexvec],
        ub_global=full_ub_global[indexvec],
        lb_hard=full_lb_hard[indexvec],
        ub_hard=full_ub_hard[indexvec],
        labels=full_labels[indexvec],
        momdat=mom,
        mmomdat=mmom,
        W=Wmat,
        mdifrec=mdifr)
end

"""
$(TYPEDSIGNATURES)

Given the model name prepares a string corresponding to the default moments to be matched in [`estimation`](@ref).

Defaults to empty string when no mode-specific method is defined.
"""
function ftypemom(mode::EstimationMode, modelname::String)

    if modelname == ""
        return ""
    else
        throw(error("this modelname is undefined"))
    end

end

"""
$(TYPEDSIGNATURES)

Set up default weighting matrix to compute the objective function. Should be defined for any subtype of [`EstimationMode`](@ref) else returns unitary matrix.
"""
function default_weight_matrix(mode::EstimationMode, typemom::String, n::Integer)
    return diagm(0 => ones(n))
end

default_weight_matrix(estset::EstimationSetup, n::Integer) = default_weight_matrix(estset.mode, estset.typemom, n)

"""
$(TYPEDSIGNATURES)

Compute moments from the data. Should be defined for any subtype of [`EstimationMode`](@ref).
"""
function datamoments(mode::EstimationMode, typemom::String)
    throw(error("a separate method has to be written for $(typeof(mode))"))
end

"""
$(TYPEDSIGNATURES)

Returns five vectors:
- `full_labels` contains the list of names of parameters which are possibly estimated under the current mode.
- `full_lb_hard` contains a hard lower bound for each parameters. If any of these bounds are violated in local stage, model moments are not computed, but instead a penalty term is returned as objective function value. For formula see [`solve_inner`](@ref). Setting values to -Inf corresponds to no lower bound.
- `full_lb_global` and 
- `full_ub_global` are lower and upper bounds for setting up the Sobol sequence. These vectors define the parameter subspace investigated in the global part of estimation. 
- `full_ub_hard` contains a hard upper bound for each parameters. Same as full_lb_hard applies. Setting values to Inf corresponds to no upper bound.

Should be defined for any subtype of [`EstimationMode`](@ref).
"""
function parambounds(mode::EstimationMode)
    throw(error("a separate method has to be written for $(typeof(mode))"))
end

"""
$(TYPEDSIGNATURES)

Create a logical vector selecting the parameters which are actually estimated from the possible full list (see full_labels in [`parambounds`](@ref)). If the `i`th element of the resulting vector is `true`, then the `i`th parameter from `full_labels` will be estimated. 

When for an EstimationMode a matching method is not separately defined, it assumes all possible parameters are always estimated and thus defaults to a vector of `true` values.
"""
function indexvector(mode::EstimationMode, modelname::String)
    return fill(true, length(parambounds(mode)[1]))
end

## GENERAL ESTIMATION ALGORITHM

"""
$(TYPEDSIGNATURES)

Perform estimation routine.
"""
function matchmom(estset::EstimationSetup, pmm::ParMM, npmm::NumParMM, cs::ComputationSettings, aux::AuxiliaryParameters, presh::PredrawnShocks, xlocstart::Array{Vector{Float64},1}, saving_bestmodel::Bool, number_bestmodel::Integer, filename_suffix::String, errorcatching::Bool)
    @unpack sobolinds, Nloc, onlyglo, onlyloc = npmm
    @unpack lb_global, ub_global = pmm
    @unpack mode, modelname, typemom = estset

    Nglo = length(sobolinds)

    onlyloc && onlyglo && throw(error("should do either local or global or both"))

    if !onlyloc
        # global stage: evaluates the objective at Sobol sequence points
        s = SobolSeq(lb_global, ub_global)
        xg0 = [Sobol.next!(s) for i in 1:sobolinds[end]]
        xg = xg0[sobolinds]

        if cs.num_procs==1 && cs.location == "local"
            objg = fill(-1.0, Nglo)
            momg = Array{Float64}(undef, length(pmm.momdat), Nglo)
            chunk_proc = 1:1:Nglo
            if cs.num_tasks==1
                singlethread_global!(objg, momg, estset, xg, pmm, aux, presh, errorcatching, chunk_proc)
            else
                multithread_global!(objg, momg, estset, xg, pmm, aux, presh, errorcatching, cs, chunk_proc)
            end            
        else
            addprocs(cs)
            load_on_procs(estset.mode)

            objg = distribute(fill(-1.0, Nglo))
            momg = distribute(Array{Float64}(undef,length(pmm.momdat), Nglo), dist = [1,cs.num_procs])

            @sync @distributed for i in eachindex(workers())
                chunk_proc = localindices(objg)[1]
                chunk_proc == localindices(momg)[2] || @error("non-conforming indices")
                
                if cs.num_tasks==1
                    singlethread_global!(objg, momg, estset, xg, pmm, aux, presh, errorcatching, chunk_proc)
                else
                    multithread_global!(objg, momg, estset, xg, pmm, aux, presh, errorcatching, cs, chunk_proc)
                end
            end
        end

        objg = Array(objg)
        momg = Array(momg)

        permg = sortperm(objg)

        objg_sort = objg[permg]
        momg_sort = [momg[:,i] for i in axes(momg,2)][permg]
        xg_sort = xg[permg]

        if cs.num_procs > 1 || cs.location == "slurm"
           @everywhere begin
               objg = nothing
               momg = nothing
           end

           rmprocs(workers())
        end

        if onlyglo

            if saving_bestmodel
                for i in 1:number_bestmodel
                    obj_mom(mode, xg_sort[i], modelname, typemom, aux, presh; saving_model=saving_bestmodel, filename=estimation_name(estset, npmm, filename_suffix) * "_$(i)")
                end
            end

            return EstimationResult(npmm, aux, presh, xlocstart, pmm,
                objg_sort, xg_sort, momg_sort,
                [0.0], [[0.0]], [[0.0]], [false])
        end

        xlocstart = xg_sort[1:Nloc]
    else
        Nloc = length(xlocstart)
    end

    # local stage: applies local optimization algorithm to find optimum starting from best global points
    if cs.num_procs==1 && cs.location == "local"
        objl = fill(-1.0, Nloc)
        moml = Array{Float64}(undef, length(pmm.momdat), Nloc)
        conv = Vector{Bool}(undef, Nloc)
        xl = Array{Float64}(undef, length(xlocstart[1]), Nloc)
        chunk_procl = 1:1:Nloc
        if cs.num_tasks == 1
            singlethread_local!(objl, xl, moml, conv, estset, npmm, pmm, aux, presh, xlocstart, errorcatching, chunk_procl)
        else
            multithread_local!(objl, xl, moml, conv, estset, npmm, pmm, aux, presh, xlocstart, errorcatching, cs, chunk_procl)
        end
    else
        addprocs(cs)
        load_on_procs(estset.mode)

        objl = distribute(fill(-1.0, Nloc))
        moml = distribute(Array{Float64}(undef,length(pmm.momdat), Nloc), dist = [1,cs.num_procs])
        conv = distribute(Vector{Bool}(undef,Nloc))
        xl = distribute(Array{Float64}(undef,length(xlocstart[1]), Nloc), dist = [1,cs.num_procs])

        @sync @distributed for i in eachindex(workers())
            chunk_procl = getchunk(1:Nloc, i; n=cs.num_procs)

            chunk_procl = localindices(objl)[1]
            chunk_procl == localindices(moml)[2] || @error("non-conforming indices")
            chunk_procl == localindices(conv)[1] || @error("non-conforming indices")
            chunk_procl == localindices(xl)[2] || @error("non-conforming indices")

            if cs.num_tasks == 1
                singlethread_local!(objl, xl, moml, conv, estset, npmm, pmm, aux, presh, xlocstart, errorcatching, chunk_procl)
            else
                multithread_local!(objl, xl, moml, conv, estset, npmm, pmm, aux, presh, xlocstart, errorcatching, cs, chunk_procl)
            end
        end
    end

    objl = Array(objl)
    moml = Array(moml)
    conv = Array(conv)
    xl = Array(xl)

    perml = sortperm(objl)

    objl_sort = objl[perml]
    moml_sort = [moml[:, i] for i in axes(moml, 2)][perml]
    xl_sort = [xl[:, i] for i in axes(xl, 2)][perml]
    conv_sort = conv[perml]

    if cs.num_procs > 1 || cs.location == "slurm"
        @everywhere begin
            objl = nothing
            moml = nothing
            conv = nothing
            xl = nothing
        end

        rmprocs(workers())
    end


    if saving_bestmodel
        for i in 1:number_bestmodel
            obj_mom(mode, xl_sort[i], modelname, typemom, aux, presh; saving_model=saving_bestmodel, filename=estimation_name(estset, npmm, filename_suffix) * "_$(i)")
        end
    end

    if onlyloc
        objg_sort = [-1.0]
        xg_sort = [[1.0]]
        momg_sort = [[1.0]]
    end

    return EstimationResult(npmm, aux, presh, xlocstart[perml], pmm,
        objg_sort, xg_sort, momg_sort,
        objl_sort, xl_sort, moml_sort, conv_sort)
end

"""
$(TYPEDSIGNATURES)

Add processes locally or on cluster.
"""
function Distributed.addprocs(cs::ComputationSettings)
    exefl = ["--project", "--threads=$(cs.num_threads)"]
    if cs.maxmem != -1
        push!(exefl,"--heap-size-hint=$(cs.maxmem)G")
    end
    if cs.location == "local"
        return addprocs(cs.num_procs, exeflags=exefl)
    elseif cs.location == "slurm"
        return addprocs(SlurmManager(cs.num_procs); exeflags=exefl, cs.clustermanager_settings...)
    else
        throw(error("cs.location has to be either 'local' or 'slurm'."))
    end
end

"""
$(TYPEDSIGNATURES)

Performs the global stage on a single-thread.
"""
function singlethread_global!(objg::AbstractVector, momg::AbstractMatrix, estset::EstimationSetup, xg::AbstractVector, pmm::ParMM, aux::AuxiliaryParameters, presh::PredrawnShocks, errorcatching::Bool, chunk_proc::AbstractVector)
    momnormg = Vector{Float64}(undef, length(pmm.momdat))
    preal = PreallocatedContainers(estset, aux)
    for (locind, fullind) in enumerate(chunk_proc)
        localpart(objg)[locind] = objf!(view(localpart(momg),:,locind), momnormg, estset, xg[fullind], pmm, aux, presh, preal, errorcatching)
    end
end

"""
$(TYPEDSIGNATURES)

Performs the global stage with multiple threads.
"""
function multithread_global!(objg::AbstractVector, momg::AbstractMatrix, estset::EstimationSetup, xg::AbstractVector, pmm::ParMM, aux::AuxiliaryParameters, presh::PredrawnShocks, errorcatching::Bool, cs::ComputationSettings, chunk_proc::AbstractVector)
    chunks_th = chunks(chunk_proc; n = cs.num_tasks)
    #prog = Progress(Nglo; desc="Performing global stage...")
    tasks = map(chunks_th) do chunk
        # Each chunk gets its own spawned task that does its own local, sequential work. every task is assigned to one thread.

        Threads.@spawn begin
            # preallocate once per chunk
            objg_ch = Vector{Float64}(undef, length(chunk))
            momg_ch = Array{Float64}(undef, length(pmm.momdat),length(chunk))
            momnormg_ch = Vector{Float64}(undef, length(pmm.momdat))
            preal = PreallocatedContainers(estset, aux)
            for n in eachindex(chunk) # do stuff for all index in chunk
                fullind = chunk_proc[chunk[n]]
                objg_ch[n] = objf!(view(momg_ch,:,n), momnormg_ch, estset, xg[fullind], pmm, aux, presh, preal, errorcatching)
                #ProgressMeter.next!(prog)
            end
            # and then returns the result
            return objg_ch, momg_ch
        end
    end
    outstates = fetch.(tasks) # collect results from tasks
    #finish!(prog)

    for (i, chunk) in enumerate(chunks_th) # organize results in final form
        localpart(objg)[chunk] = outstates[i][1]
        localpart(momg)[:, chunk] = outstates[i][2]
    end
end

"""
$(TYPEDSIGNATURES)

Performs the local stage on a single-thread.
"""
function singlethread_local!(objl::AbstractVector, xl::AbstractMatrix, moml::AbstractMatrix, conv::AbstractVector, estset::EstimationSetup, npmm::NumParMM, pmm::ParMM, aux::AuxiliaryParameters, presh::PredrawnShocks, xcands::AbstractVector, errorcatching::Bool, chunk_procl::AbstractVector)
    momnorml = Vector{Float64}(undef, length(pmm.momdat))
    preal = PreallocatedContainers(estset, aux)
    for (locind, fullind) in enumerate(chunk_procl)
        opt_loc!(localpart(objl), localpart(xl), localpart(moml), momnorml, localpart(conv), npmm.local_opt_settings, estset, aux, presh, preal, pmm, xcands[fullind], locind, errorcatching)
        objf!(view(localpart(moml), :, locind), momnorml, estset, localpart(xl)[:, locind], pmm, aux, presh, preal, errorcatching)
    end
end

"""
$(TYPEDSIGNATURES)

Performs the local stage with multiple threads.
"""
function multithread_local!(objl::AbstractVector, xl::AbstractMatrix, moml::AbstractMatrix, conv::AbstractVector, estset::EstimationSetup, npmm::NumParMM, pmm::ParMM, aux::AuxiliaryParameters, presh::PredrawnShocks, xcands::AbstractVector,errorcatching::Bool, cs::ComputationSettings, chunk_procl::AbstractVector)
    chunks_th = chunks(chunk_procl; n=cs.num_tasks)
    #prog = Progress(Nloc; desc="Performing local stage...")
    tasks = map(chunks_th) do chunk
        Threads.@spawn begin
            objl_ch = Vector{Float64}(undef, length(chunk))
            xl_ch = Array{Float64}(undef, size(xl,1), length(chunk))
            moml_ch = Array{Float64}(undef, length(pmm.momdat), length(chunk))
            conv_ch = Array{Bool}(undef, length(chunk))
            momnorml_ch = Vector{Float64}(undef, length(pmm.momdat))
            preal = PreallocatedContainers(estset, aux)
            for n in eachindex(chunk)
                fullind = chunk_procl[chunk[n]]
                opt_loc!(objl_ch, xl_ch, moml_ch, momnorml_ch, conv_ch, npmm.local_opt_settings, estset, aux, presh, preal, pmm, xcands[fullind], n, errorcatching)
                objf!(view(moml_ch, :,n), momnorml_ch, estset, xl_ch[:, n], pmm, aux, presh, preal, errorcatching)
                #ProgressMeter.next!(prog)
            end
            return objl_ch, xl_ch, moml_ch, conv_ch
        end
    end
    outstates = fetch.(tasks)
    #finish!(prog)
    for (i, chunk) in enumerate(chunks_th)
        localpart(objl)[chunk] = outstates[i][1]
        localpart(xl)[:,chunk] = outstates[i][2]
        localpart(moml)[:,chunk] = outstates[i][3]
        localpart(conv)[chunk] = outstates[i][4]
    end
end

"""
$(TYPEDSIGNATURES)

Perform estimation routine, local stage.
"""
function opt_loc!(obj::AbstractVector, xsol::AbstractMatrix, mom::AbstractMatrix, momnorm::AbstractVector, conv::AbstractVector, local_opt_settings::Dict{Symbol,Any}, estset::EstimationSetup, aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers, pmm::ParMM, xcand::Vector{Float64}, n::Int64, errorcatching::Bool)
    problem = OptimizationProblem((y, unused) -> objf!(view(mom, :, n), momnorm, estset, y, pmm, aux, presh, preal, errorcatching), xcand)
    settings = deepcopy(local_opt_settings)
    algorithm = pop!(settings,:algorithm)
    solution = solve(problem, algorithm; settings...)
    obj[n] = solution.objective
    xsol[:, n] = solution.u
    conv[n] = SciMLBase.successful_retcode(solution.retcode)

    return nothing
end

## AUXILIARY FUNCTIONS 

"""
$(TYPEDSIGNATURES)

Computes the objective function.
"""
function objf!(mom::AbstractVector, momnorm::AbstractVector, estset::EstimationSetup, x::AbstractVector, pmm::ParMM, aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers, errorcatching::Bool)
    @unpack lb_hard, ub_hard, W, momdat, mmomdat, mdifrec = pmm
    @unpack mode, modelname, typemom = estset
    flat_penalty = 10^15

    if !all(x .>= lb_hard) || !all(x .<= ub_hard) # return penalty if outside bounds
        # println("guess is outside of feasible region: $x")
        return flat_penalty + norm(max.(x .- ub_hard, [0.0])) + norm(max.(lb_hard .- x, [0.0]))
    else
        if errorcatching
            try
                _objf!(mom, momnorm, estset, x, pmm, aux, presh, preal)
            catch e
                println("unfortunate error with parameter vector $x")
                @error "ERROR: " exception = (e, catch_backtrace())
                return flat_penalty # return penalty if error
            end
        else
            _objf!(mom, momnorm, estset, x, pmm, aux, presh, preal)
        end
    end
end

function _objf!(mom::AbstractVector, momnorm::AbstractVector, estset::EstimationSetup, x::Vector{Float64}, pmm::ParMM, aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers)
    @unpack W, momdat, mmomdat, mdifrec = pmm
    @unpack mode, modelname, typemom = estset


    obj_mom!(mom, momnorm, mode, x, modelname, typemom, aux, presh, preal) # obtains moments from model

    mdif = mdiff(mode, mom, momdat, mmomdat) .- mdifrec # computes deviation from moments in data

    traditional_obj = mdif' * W * mdif

    return sqrt(traditional_obj / tr(W))


end

"""
$(TYPEDSIGNATURES)

Routine to compute moments in model under a given parameter guess. Should be defined for any subtype of [`EstimationMode`](@ref).
"""
function obj_mom!(mom::AbstractVector, momnorm::AbstractVector, mode::EstimationMode, x::Vector{Float64}, modelname::String, typemom::String, aux::AuxiliaryParameters, presh::PredrawnShocks, preal::PreallocatedContainers; saving_model::Bool=false, filename::String="")
    throw(error("a separate method has to be written for $(typeof(mode))"))
end

function obj_mom(mode::EstimationMode, x::Vector{Float64}, modelname::String, typemom::String, aux::AuxiliaryParameters, presh::PredrawnShocks; saving_model::Bool=false, filename::String="")
    momlen = length(momentnames(mode, typemom)[!, 1])
    mom = Vector{Float64}(undef, momlen)
    momnorm = Vector{Float64}(undef, momlen)
    obj_mom!(mom, momnorm, mode, x, modelname, typemom, aux, presh, PreallocatedContainers(mode, modelname, typemom, aux); saving_model, filename)
    return mom
end

"""
$(TYPEDSIGNATURES)

Computes deviation of model moments from data moments. Should be defined for any subtype of [`EstimationMode`](@ref).

Default: deviation is obtained by rescaling differences with data means of respective quantities.
"""
function mdiff(mode::EstimationMode, m::AbstractVector, momdat::AbstractVector, mmomdat::AbstractVector)
    return (momdat .- m) ./ mmomdat
end

"""
$(TYPEDSIGNATURES)

Loads model-specific files on all processes. Used only with multiprocessing. Should be defined for any subtype of [`EstimationMode`](@ref).
"""
function load_on_procs(mode::EstimationMode)
    throw(error("A separate method of 'load_on_procs' has to be written for $(typeof(mode)) when using multiprocessing."))
end

# OTHER USEFUL FUNCTIONS

"""
$(TYPEDSIGNATURES)

List of estimated parameters. 
"""
function labels(estset::EstimationSetup)
    @unpack mode, modelname = estset
    full_labels = parambounds(mode)[1]
    indexv = indexvector(mode, modelname)
    return full_labels[indexv]
end

"""
$(TYPEDSIGNATURES)

Saving required elements. 
"""
function save_estimation(estset::EstimationSetup, npmm::NumParMM, mmsolu::EstimationResult, filename_suffix::String)

    filename = estimation_result_path() * estimation_name(estset, npmm, filename_suffix) * ".jld"

    save(filename, "mmsolu", mmsolu, "estset", estset)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Saving required elements, lighter version to save memory. 
"""
function save_estimation_lightweight(estset::EstimationSetup, mmsolu::EstimationResult; filename_suffix::String="", filename::String=estimation_name(estset, mmsolu.npmm, filename_suffix), bestN::Integer=5000)

    mmsolu2 = EstimationResult(mmsolu.npmm, mmsolu.aux, EmptyPredrawnShocks(), mmsolu.xlocstart, mmsolu.pmm, mmsolu.fglo[1:min(bestN, end)], mmsolu.xglo[1:min(bestN, end)], mmsolu.momglo[1:min(bestN, end)], mmsolu.floc[1:min(bestN, end)], mmsolu.xloc[1:min(bestN, end)], mmsolu.momloc[1:min(bestN, end)], mmsolu.conv[1:min(bestN, end)])

    filename = estimation_result_path() * filename * "_lightweight" * ".jld"
    save(filename, "mmsolu", mmsolu2, "estset", estset)
    return nothing
end


function save_estimation_lightweight(filename::String; path::String=estimation_result_path(), bestN::Integer=1000)
    res = load(path * filename * ".jld")
    estset = res["estset"]
    mmsolu = res["mmsolu"]

    return save_estimation_lightweight(estset, mmsolu; filename, bestN)
end

"""
$(TYPEDSIGNATURES)

Provide middle part of filename, showing which kind of estimation is done.
"""
function estimation_name(estset::EstimationSetup, npmm::NumParMM, filename_suffix::String)
    @unpack mode, modelname, typemom = estset
    @unpack onlyloc, onlyglo = npmm

    if onlyloc == true
        lgstr = "_ol"
    elseif onlyglo == true
        lgstr = "_og"
    else
        lgstr = ""
    end

    modelnamest = modelname == "" ? "" : "_" * modelname
    typemomst = typemom == "" ? "" : "_" * typemom

    return mode.filename * modelnamest * typemomst * lgstr * filename_suffix
end

estimation_name(estset::EstimationSetup, filename_suffix::String) = estimation_name(estset, NumParMM(estset), filename_suffix)

estimation_name(estset::EstimationSetup) = estimation_name(estset, "")

"""
$(TYPEDSIGNATURES)

Returns the full names of the moments. Should be defined for any subtype of [`EstimationMode`](@ref).
"""
function momentnames(mode::EstimationMode, typemom::String)
    throw(error("a separate method has to be written for $(typeof(mode))"))
end

momentnames(estset::EstimationSetup) = momentnames(estset.mode, estset.typemom)

"""
$(TYPEDSIGNATURES)

Two-step estimation procedure.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- npmomsim: Contains the numerical parameters used to simulate the distribution of moments when model and the estimated parameters are assumed to be correct. Its format is not restricted, but `defaultInnerAuxStructure(mode,npmomsim)` should return a valid aux structure. Should imitate the process generating the data, based on which the original moments were computed. 
- nprepeat: Contains the numerical parameters used in simulation when reestimating the parameters of the model with alternative moments to match and different seeds. Its format is not restricted, but `defaultInnerAuxStructure(mode,npmomsim)` should return a valid aux structure. Should imitate the setup of the original estimation.
- Nseeds: Number of different seeds to try for each alternative sample moment vector.
- Nsamplesim: Number of alternative sample moment vectors to try.
- Ndata: Size of data sample.

# Optional arguments
- npmm: Numerical parameters for estimation. See separate documentation [`NumParMM`](@ref).
- saving_est: Logical, true if estimation results are to be saved.
- saving_bestmodel: Logical, if some object linked to model solution corresponding to the best parameters are to be saved.
- filename_suffix: String with suffix to be used for file name when saving.
"""
function two_stage_estimation(estset::EstimationSetup, auxmomsim::AuxiliaryParameters, Nseeds::Integer, Nsamplesim::Integer, Ndata::Integer; aux::AuxiliaryParameters=AuxiliaryParameters(estset), npmm::NumParMM=NumParMM(estset.mode), saving::Bool=true, saving_bestmodel::Bool=saving, filename_suffix::String="", errorcatching::Bool=false)

    est_1st = estimation(estset; aux, npmm, saving, saving_bestmodel, filename_suffix=filename_suffix * "_1st", errorcatching)
    boot_1st = param_bootstrap_result(estset, est_1st, auxmomsim, Nseeds, Nsamplesim, Ndata; saving, filename_suffix=filename_suffix * "_1st")

    est_2st = estimation(estset; aux, npmm, saving, saving_bestmodel, filename_suffix=filename_suffix * "_2st", Wmat=boot_1st.W, errorcatching)
    boot_2st = param_bootstrap_result(estset, est_2st, auxmomsim, Nseeds, Nsamplesim, Ndata; saving, filename_suffix=filename_suffix * "_2st")

    return est_1st, boot_1st, est_2st, boot_2st
end

macro maythread(body)
    esc(:(
        if $(@__MODULE__).threading_inside()
            Threads.@threads($body)
        else
            $body
        end
    ))
end

estimation_result_path() = "./saved/estimation_results/"

threading_inside() = false
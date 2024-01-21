module MomentMatching

# Dependencies
# we might want to specify which functions we need and import only those

using Plots # for functions providing figures. Should check if we can avoid via recipes. Calling Plot in a library is not adviced, since people use different back-ends.
using Plots: plot, plot! # importing these functions explicitly avoids bug in vscode underlying all plot functions. Should check if still needed or has been fixed.
gr()

using LinearAlgebra: diagm, tr, norm, cond, diag
#using Statistics # let's try if we get away with commenting this out. StatsBase is bigger and calls many functions from Statistics anyways.
using StatsBase: mean, var, cov
using JLD # For saving results. We should consider alternatives due to compatibility issues
using Parameters: @with_kw, @unpack # For keywords in types
using DocStringExtensions: FIELDS, TYPEDSIGNATURES, TYPEDEF # For easier documentation. Should we use it in the end?
using CSV # For saving tables as output
using DataFrames # For dealing with tables
using Optim # For global phase of estimation
using Sobol # For global phase of estimation
using ProgressMeter # For showing progress while running long computations

# exported types and functions
export estimation, # main functions
       two_stage_estimation,

       # from estimation.jl:
       # Abstract types and functions which (might) need 
       # mode-specific subtypes or methods.
       AuxiliaryParameters,
       EstimationMode,
       PreallocatedContainers,       
       PredrawnShocks,

       datamoments,
       default_weight_matrix,
       ftypemom,
       indexvector,
       mdiff,
       momentnames,
       obj_mom,
       parambounds,

       # from estimation.jl:
       # other objects which must be directly available for the user
       BootstrapResult,
       EmptyPredrawnShocks,
       estimation_name,
       estimation_output_path,
       estimation_result_path,
       EstimationResult,
       EstimationSetup,
       @maythread,
       NumParMM,
       ParMM,
       threading_inside,

       # from inference.jl:
       # functions to perform inference on results
       marginal_fobj,
       defaultxgrid,
       param_bootstrap_result,
       Jtest,

       # from output.jl:
       # functions to generate figures or tables
       alloutputs,
       fbootstrap,
       fglobounds,
       fmarg,
       fmoms,
       fsanity,
       tableest,
       tablemoms

include("estimation.jl")
include("inference.jl")
include("output.jl")

end

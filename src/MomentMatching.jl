module MomentMatching

# Dependencies
# we might want to specify which functions we need and import only those

using Plots # for functions providing figures. Should check if we can avoid via recipes. Calling Plot in a library is not adviced, since people use different back-ends.
using Plots: plot, plot! # importing these functions explicitly avoids bug in vscode underlying all plot functions. Should check if still needed or has been fixed.
gr()

# using LinearAlgebra # sure?
#using Statistics # let's try if we get away with commenting this out. StatsBase is bigger and calls many functions from Statistics anyways.
#using StatsBase # sure?
using JLD # For saving results. We should consider alternatives due to compatibility issues
using Parameters: @with_kw, @unpack # For keywords in types
using DocStringExtensions: FIELDS, TYPEDSIGNATURES, TYPEDEF # For easier documentation. Should we use it in the end?
using CSV # For saving tables as output
using DataFrames # For dealing with tables
using Optim # For global phase of estimation
using Sobol # For global phase of estimation
using ProgressMeter # For showing progress while running long computations

# exported types and functions
export estimation, # main function
       two_stage_estimation


include("estimation.jl")
include("inference.jl")
include("output.jl")

end
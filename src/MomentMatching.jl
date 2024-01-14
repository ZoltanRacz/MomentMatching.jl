module MomentMatching

# Write your package code here.

using Plots
using Plots: plot, plot! # importing these functions explicitly avoids bug in vscode underlying all plot functions 
gr()
using Plots.PlotMeasures
using LinearAlgebra
#using Statistics # let's try if we get away with commenting this out. StatsBase is bigger and calls many functions from Statistics anyways.
#using StatsBase
using JLD 
using Parameters: @with_kw, @unpack
using DocStringExtensions: FIELDS, TYPEDSIGNATURES, TYPEDEF
using CSV
using DataFrames
#using Interpolations
using Optim
using Sobol
#using FastGaussQuadrature
#using Roots
#using SparseArrays
#using Random
#using Distributions
using ProgressMeter
using Distances: euclidean

include("estimation.jl")
include("inference.jl")
include("output.jl")

export estimation,
       two_stage_estimation


end

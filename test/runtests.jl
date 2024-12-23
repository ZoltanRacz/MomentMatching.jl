using MomentMatching
using Test

include("examples/minimalAR1.jl")

test_multiprocessing = false

test_multiprocessing || @warn("Multiprocessing is not tested.")

@testset "Unit test of global estimation" begin
    include("unittest_globalestimation.jl")
end

@testset "Unit test of local estimation" begin
    include("unittest_localestimation.jl")
end

@testset "Unit test of bootstrap" begin
    include("unittest_bootstrap.jl")
end

@testset "Unit test of functions to merge results" begin
    include("unittest_mergeresults.jl")
end

@testset "Integration test of minimal example" begin
    include("integrationtest_minimal.jl")
end
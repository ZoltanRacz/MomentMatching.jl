using MomentMatching
using Test

@testset "Unit test of global estimation" begin
    include("unittest_estimation.jl")
end

@testset "Integration test of minimal example" begin
    include("integrationtest_minimal.jl")
end
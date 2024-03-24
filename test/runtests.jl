using MomentMatching
using Test

@testset "Unit test of global estimation" begin
    include("unittest_globalestimation.jl")
end

@testset "Unit test of local estimation" begin
    include("unittest_localestimation.jl")
end

@testset "Integration test of minimal example" begin
    include("integrationtest_minimal.jl")
end
# useful lines for testing manually, while developing. Install TestEnv in your main environment. When running the first time, activate and instantiate the test environment before restarting Julia and using TestEnv. For more info check: https://github.com/JuliaTesting/TestEnv.jl/blob/main/README.md
#using TestEnv
#TestEnv.activate()
using MomentMatching, Test, Plots, OptimizationOptimJL

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npest = NumParMM(setup; Nglo=100, onlyglo = true)

Threads.nthreads() != 1 || @warn("multithreading is not tested")

# we need common shocks to test equivalence of methods
aux = AuxiliaryParameters(AR1Estimation("ar1estim"), "")
presh = PredrawnShocks(AR1Estimation("ar1estim"), "", "", aux) 

cp11 = ComputationSettings(num_procs = 1, num_tasks = 1)
est_11 = estimation(setup; npmm=npest, cs = cp11, presh = presh, saving=false)
@test est_11 isa EstimationResult

cp14 = ComputationSettings(num_procs = 1, num_tasks = 4)
est_14 = estimation(setup; npmm=npest, cs = cp14, presh = presh, saving=false)
@test est_14 isa EstimationResult

cp31 = ComputationSettings(num_procs = 3, num_tasks = 1)
est_31 = estimation(setup; npmm=npest, cs = cp31, presh = presh, saving=false)
@test est_31 isa EstimationResult

cp34 = ComputationSettings(num_procs = 3, num_tasks = 4)
est_34 = estimation(setup; npmm=npest, cs = cp34, presh = presh, saving=false)
@test est_34 isa EstimationResult

@test est_11.fglo == est_14.fglo && est_11.xglo == est_14.xglo && est_11.momglo == est_14.momglo
@test est_11.fglo == est_31.fglo && est_11.xglo == est_31.xglo && est_11.momglo == est_31.momglo
@test est_11.fglo == est_34.fglo && est_11.xglo == est_34.xglo && est_11.momglo == est_34.momglo
# useful lines for testing manually, while developing. Install TestEnv in your main environment. When running the first time, activate and instantiate the test environment before restarting Julia and using TestEnv. For more info check: https://github.com/JuliaTesting/TestEnv.jl/blob/main/README.md
#using TestEnv
#TestEnv.activate()
using MomentMatching, Test, OptimizationOptimJL
using Distributed

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

nloc = 12
npmm = NumParMM(setup; Nloc = nloc, onlyloc = true, local_opt_settings = (algorithm = NelderMead(), maxiters = 50))

xlocstart = [
[0.7,0.01,0.01], [0.8,0.01,0.01], [0.9,0.01,0.01], 
[0.7,0.02,0.01], [0.8,0.02,0.01], [0.9,0.02,0.01], 
[0.7,0.01,0.02], [0.8,0.01,0.02], [0.9,0.01,0.02], 
[0.7,0.02,0.02], [0.8,0.02,0.02], [0.9,0.02,0.02]]

Threads.nthreads() != 1 || @warn("multithreading is not tested")

# we need common shocks to test equivalence of methods
aux = AuxiliaryParameters(AR1Estimation("ar1estim"), "")
presh = PredrawnShocks(AR1Estimation("ar1estim"), "", "", aux) 

cs_11 = ComputationSettings(num_procs = 1, num_tasks = 1)
est_11 = estimation(setup; npmm, presh, xlocstart, cs = cs_11, saving=false)
@test est_11 isa EstimationResult

cs_14 = ComputationSettings(num_procs = 1, num_tasks = 4)
est_14 = estimation(setup; npmm, presh, xlocstart, cs = cs_14, saving=false)
@test est_14 isa EstimationResult

cs_31 = ComputationSettings(num_procs = 3, num_tasks = 1)
est_31 = estimation(setup; npmm, presh, xlocstart, cs = cs_31, saving=false)
@test est_31 isa EstimationResult

cs_34 = ComputationSettings(num_procs = 3, num_tasks = 4)
est_34 = estimation(setup; npmm, presh, xlocstart, cs = cs_34, saving=false)
@test est_34 isa EstimationResult

plot(est_31.fglo)
plot!(est_11.fglo)
plot!(est_14.fglo)
plot!(est_34.fglo)

@test est_11.fglo == est_14.fglo && est_11.xglo == est_14.xglo && est_11.momglo == est_14.momglo
@test est_11.fglo == est_31.fglo && est_11.xglo == est_31.xglo && est_11.momglo == est_31.momglo
@test est_11.fglo == est_34.fglo && est_11.xglo == est_34.xglo && est_11.momglo == est_34.momglo
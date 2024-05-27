# useful lines for testing manually, while developing. Install TestEnv in your main environment. When running the first time, activate and instantiate the test environment before restarting Julia and using TestEnv. For more info check: https://github.com/JuliaTesting/TestEnv.jl/blob/main/README.md
#using TestEnv
#TestEnv.activate()
using MomentMatching, Test
using OptimizationOptimJL
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
est = estimation(setup; npmm, presh, xlocstart, cs = cs_11, saving=false)

Tdis = 20
Ndata = 500
Tdata = 40
Nsample = 4
Nseed = 4
auxmomsim = AR1AuxPar(Ndata, Tdata + Tdis, Tdis)

boot_11 = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false, cs = cs_11)
@test boot_11 isa BootstrapResult

cs_14 = ComputationSettings(num_procs = 1, num_tasks = 4)
boot_14 = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false, cs = cs_14)
@test boot_14 isa BootstrapResult

cs_31 = ComputationSettings(num_procs = 3, num_tasks = 1)
boot_31 = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false, cs = cs_31)
@test boot_31 isa BootstrapResult

cs_34 = ComputationSettings(num_procs = 3, num_tasks = 4)
boot_34 = param_bootstrap_result(setup, est, auxmomsim, Nseed, Nsample, Ndata, saving=false, cs = cs_34)
@test boot_34 isa BootstrapResult


using Plots
fbootstrap(setup, est, boot_11)
fbootstrap(setup, est, boot_14)
fbootstrap(setup, est, boot_31)
fbootstrap(setup, est, boot_34)
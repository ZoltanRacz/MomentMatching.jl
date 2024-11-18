# useful lines for testing manually, while developing. Install TestEnv in your main environment. When running the first time, activate and instantiate the test environment before restarting Julia and using TestEnv. For more info check: https://github.com/JuliaTesting/TestEnv.jl/blob/main/README.md
#using TestEnv
#TestEnv.activate()
using MomentMatching, Test, Plots, OptimizationOptimJL

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npest = NumParMM(setup; Nglo=100, Nloc=10, local_opt_settings = (algorithm = NelderMead(), maxtime = 30.0))

Tdis = 20
Ndata = 500
Tdata = 40
Nsample = 4
Nseed = 4
auxmomsim = AR1AuxPar(Ndata, Tdata + Tdis, Tdis)

aux = AuxiliaryParameters(AR1Estimation("ar1estim"), "")
presh = PredrawnShocks(AR1Estimation("ar1estim"), "", "", aux)
preal = PreallocatedContainers(AR1Estimation("ar1estim"), "", "", aux)

@test aux isa AR1AuxPar
@test presh isa AR1PreShocks
@test preal isa AR1PrealCont

mom = fill(0.0, 3)
momn = fill(0.0, 3)
MomentMatching.obj_mom!(mom, momn, AR1Estimation("ar1estim"), [0.9, 0.2, 0.1], "", "", aux, presh, preal)
@test mom[1] != 0.0
@test momn[1] != 0.0

est_1st = estimation(setup; npmm=npest, saving=false)
@test est_1st isa EstimationResult

boot_1st = param_bootstrap_result(setup, est_1st, auxmomsim, Nseed, Nsample, Ndata, saving=false)
@test boot_1st isa BootstrapResult

marg = marginal_fobj(setup, est_1st, 17, fill(0.1, 3))
fmarg(setup, est_1st, marg)

marg = marginal_fobj(setup, est_1st, 17, fill(0.1, 3), which_point=2)
fmarg(setup, est_1st, marg, which_point=2)

fsanity(setup, est_1st)

fmoms(setup, est_1st, 1)
fmoms(setup, est_1st, 1, which_point = 10)

fmoms(setup, est_1st)
fmoms(setup, est_1st, display_all = false)
fmoms(setup, est_1st, which_point = 2)

fbootstrap(setup, est_1st, boot_1st)

flocalorder(est_1st)
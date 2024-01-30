using MomentMatching, Test

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"),"","")

npest = NumParMM(setup; Nglo=100, Nloc=10, it = 3000)

Tdis = 100
Ndata = 3000
Tdata = 300
Nsample = 3
Nseed = 2
auxmomsim = AR1AuxPar(Ndata,Tdata+Tdis,Tdis)

aux = AuxiliaryParameters(AR1Estimation("ar1estim"),"")
presh = PredrawnShocks(AR1Estimation("ar1estim"),"","",aux)
preal = PreallocatedContainers(AR1Estimation("ar1estim"),"","",aux)

@test aux isa AR1AuxPar
@test presh isa AR1PreShocks
@test preal isa AR1PrealCont

mom = fill(0.0,3)
momn = fill(0.0,3)
MomentMatching.obj_mom!(mom, momn, AR1Estimation("ar1estim"), [0.9,0.2,0.1], "", "", aux, presh, preal)
@test mom[1] != 0.0
@test momn[1] != 0.0

est_1st = estimation(setup; npmm=npest, saving = false)
@test est_1st isa EstimationResult

boot_1st = param_bootstrap_result(setup, est_1st,auxmomsim, Nseed, Nsample, Ndata, saving = false)
@test boot_1st isa BootstrapResult


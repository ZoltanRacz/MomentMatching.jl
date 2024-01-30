using MomentMatching, Test

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"),"","")

npest = NumParMM(setup; Nglo=100, Nloc=10, it = 3000)

Tdis = 100
Ndata = 30000
Tdata = 300
Nsample = 150
Nseed = 10
auxmomsim = AR1AuxPar(Ndata,Tdata+Tdis,Tdis)

aux = AuxiliaryParameters(AR1Estimation("ar1estim"),"")
presh = PredrawnShocks(AR1Estimation("ar1estim"),"","",aux)
preal = PreallocatedContainers(AR1Estimation("ar1estim"),"","",aux)

MomentMatching.obj_mom!(fill(0.0,3), fill(0.0,3), AR1Estimation("ar1estim"), [0.9,0.2,0.1], "", "", aux, presh, preal)

est_1st = estimation(setup; npmm=npest, saving = false)
boot_1st = param_bootstrap_result(setup, est_1st,auxmomsim, Nseed, Nsample, Ndata, saving = false)
using MomentMatching, Test

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"),"","")

npest = NumParMM(setup; Nglo=1000, Nloc=10, it = 3000)

Tdis = 100
Ndata = 30000
Tdata = 300
Nsample = 150
Nseed = 10
auxmomsim = AR1AuxPar(Ndata,Tdata+Tdis,Tdis)

# one stage, first only estimation (bootstrap takes long, do only for ok estimation results)
est_1st = estimation(setup; npmm=npest, filename_suffix = "_1st")
aggboot_1st = param_bootstrap_result(aggsetup, aggest_1st,auxmomsimagg, Nseed, Nsample, Ndata, saving = true, filename_suffix = "_1st")
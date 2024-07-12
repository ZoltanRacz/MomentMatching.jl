# useful lines for testing manually, while developing. Install TestEnv in your main environment. When running the first time, activate and instantiate the test environment before restarting Julia and using TestEnv. For more info check: https://github.com/JuliaTesting/TestEnv.jl/blob/main/README.md
#using TestEnv
#TestEnv.activate()
using MomentMatching, Test, OptimizationOptimJL

include("examples/minimalAR1.jl")

setup = EstimationSetup(AR1Estimation("ar1estim"), "", "")

npmm = NumParMM(setup; sobolinds = 1:100, Nloc = 12, local_opt_settings = (algorithm = NelderMead(), maxiters = 50))

# we need common shocks to test equivalence of methods
aux = AuxiliaryParameters(AR1Estimation("ar1estim"), "")
presh = PredrawnShocks(AR1Estimation("ar1estim"), "", "", aux) 

# without merging - for comparison
est = estimation(setup; npmm, presh, saving=false)

# global

npmm_glo_1 = NumParMM(setup; sobolinds = 1:40,onlyglo = true)
npmm_glo_2 = NumParMM(setup; sobolinds = 41:100,onlyglo = true)

est_glo_1 = estimation(setup; npmm = npmm_glo_1, presh, saving=false)
est_glo_2 = estimation(setup; npmm = npmm_glo_2, presh, saving=false)

est_glo = mergeglo(setup, [est_glo_1, est_glo_2], saving = false)

# local

npmm_loc_1 = NumParMM(setup; Nloc = 8, onlyloc = true, local_opt_settings = (algorithm = NelderMead(), maxiters = 50))

npmm_loc_2 = NumParMM(setup; Nloc = 4, onlyloc = true, local_opt_settings = (algorithm = NelderMead(), maxiters = 50))

est_loc_1 = estimation(setup; npmm = npmm_loc_1, presh, saving=false, xlocstart = est_glo.xglo[1:8])
est_loc_2 = estimation(setup; npmm = npmm_loc_2, presh, saving=false, xlocstart = est_glo.xglo[9:12])

est_loc = mergeloc(setup, [est_loc_1, est_loc_2], saving = false)

# merge global and local

est_merged = mergegloloc(setup, est_glo, est_loc, saving = false)

# test if identical results to estimating at once

@test est.fglo == est_merged.fglo && est.xglo == est_merged.xglo && est.momglo == est_merged.momglo

@test est.floc == est_merged.floc && est.xloc == est_merged.xloc && est.momloc == est_merged.momloc
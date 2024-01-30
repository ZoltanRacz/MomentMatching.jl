# MomentMatching

[![Build Status](https://github.com/ZoltanRacz/MomentMatching.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ZoltanRacz/MomentMatching.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/ZoltanRacz/MomentMatching.jl/graph/badge.svg?token=YLP96BUQ9S)](https://codecov.io/gh/ZoltanRacz/MomentMatching.jl)

by Gualtiero Azzalini (Stockholm School of Economics)\
Zoltan Racz (Stockholm School of Economics)

__Description.__ This package allows to perform Method of Moments estimation (Generalized and Simulated).  
It comprises three files which are not model-specific and are used in all applications and model-specific files. See below for more details.

## Routine files
There are three files which are not model-specific and which are used in all applications:
1. _SMM_GMM_estimation_: contains structures and functions to perform the estimation;
2. _SMM_GMM_inference_: contains functions to perform inference and diagnostics;
3. _SMM_GMM_output_: contains structures and functions to produce tables and figures to display the estimation results.

### 1. SMM_GMM_estimation

This is the core file performing the estimation routine. To describe the procedure let's introduce some useful notation:
- $x$: array of data; 
- $\theta$: array containing $K$ parameters to be estimated;
- $g(x, \theta)$: array of $M$ moment conditions, function of data and parameters, in the form $g(x, \theta)=0$;
- $W$: weighting matrix for the moment conditions with dimension $M \times M$; 
- $g^{\prime}Wg$: objective function to be minimized;
- $S=\mathbb{E}[g(x, \theta)g(x, \theta)^{\prime}]$: covariance matrix of the moments;
- $G$: matrix with the derivatives of the moment conditions with respect to $\theta$ with dimension $M \times K$;
- Hat over quantity: identifies estimators.

_Note_: identification requires $M \geq K$.

The estimation algorithm in the function [`estimation`](@ref) includes a _global_ and a _local_ stage in its most general form. Nevertheless, the code is flexible enough to allow performing only one of the two stages at a time. 

__Global stage.__ In the global stage (see [`matchmom`](@ref)) a Sobol sequence is generated on the parameter space defined by $\theta$. The objective function is evaluated at each combination of parameters in the sequence. If only the global stage is performed, the best (in terms of lowest value of the objective function) combination of parameters is returned as the minimizer. Else, the best points (the number can be specified by the user) are used as starting guesses for the local stage. 

_Note_: it is important that the number of global points evaluated is large, so that all the areas of the parameter space are searched well enough. This is especially true if the objective function is not well behaved.

__Local stage.__ The most promising candidates found in the global stage are then used as starting points for a local stage (see [`opt_loc!`](@ref)), in which the Nelder-Mead algorithm is used to find the minimum of the objective function. If only the local stage is performed, the initial guess can be supplied by the user. The global minimizer is then the best point among the values returned at the end of the local minimization routine at each tried starting guess.     

The file also contains a function [`two_stage_estimation`](@ref) to perform two-step GMM estimation. In the first step estimation is performed with a default weighting matrix. Parametric bootstrap (described in the next section) then allows to get the updated efficient weighting matrix (under the assumption that the model and estimated parameters are correct). The latter is then used as weighting matrix for estimation in the second step.

### 2. SMM_GMM_inference

This file contains functions to perform diagnostics and inference on the estimation results. 

__Diagnostics.__ In order to check that a minimum is achieved, the function [`marginal_fobj`](@ref) evaluates the objective function at points around the estimated parameter $\hat{\theta}$ in one dimension at a time (see below for functions to plot this check) and saves the results in an array.

__Parametric bootstrap.__ In order to gauge how precisely the targeted moments are estimated in the original data, a function [`param_bootstrap`](@ref) to perform parametric bootstrap is provided. Specifically, assuming model and estimated parameters are correct, alternative samples are simulated - with the same size as the original sample data sample - and, for each sample, an alternative vector of moments to match is computed. Then, for every alternative moment vector, the estimation procedure (local stage) is repeated. The resulting distribution of estimated parameters can be used to obtain the bootstrap confidence intervals. To check that the chosen seed does not influence the results (in which case it is recommended to restart the whole estimation with bigger simulation sizes), this procedure can optionally be repeated with several seeds.

__Standard errors.__ The asymptotic variance [`sandwich_matrix`](@ref) of the parameter estimates has the usual sandwich form: $(G^\prime W G)^{-1} G^\prime W S W G(G^\prime W G)^{-1}$. $\hat{G}$ can is computed numerically at the estimated parameter value $\hat{\theta}$ with the function [`Qmatrix`](@ref). $W$ is the weighting matrix used in estimation. Under the assumptions of the parametric bootstrap, there are two sources of variation in the parameter estimates: variation in the simulated sample for a given seed and across seeds. The parametric bootstrap procedure just outlined returns a distribution of the parameter estimates that considers exactly these two factors. Thus, the covariance matrix of the simulated moments is a consistent estimator for $S$ and, after having been computed by the function [`Omega_boots`](@ref), it is used as $\hat{S}$. 

_Note_: If the model is just identified or if $\hat{W}$ is a consistent estimator of $S^{-1}$ then the variance simplifies to $(G^\prime S^{-1} G)^{-1}$. The latter happens, for instance, when $\hat{W}$ is the efficient weighting matrix obtained in the two-stage estimation procedure or, in the parametric bootstrap case just described when $\hat{W}$ is set to the inverse of the consistent estimator of $S$.  

__J-test.__ A function [`Jtest`](@ref) to test over-identifying restriction is provided. The $J$-statistic is defined as $n\hat{g}^{\prime}\hat{W}\hat{g}$ where $\hat{W}$ is a consistent estimator of $S^{-1}$, $\hat{g}$ a consistent estimator of the moment conditions and $n$ is the sample size. Under the null $J \rightarrow_{d} \chi^2_{M-K}$.  

### 3. SMM_GMM_output

This file contains functions to organize the estimation results and to perform some sanity checks on them. More specifically, the following figures and tables can be generated: 

* [`fsanity`](@ref): figure plotting the estimated parameters in every trial of the global or local stage in increasing order with respect to the objective function value obtained. If the found optimum is stable, then the estimated parameter values should not vary much, especially for the trials that achieved objective function values close to the minimum.

* [`fmarg`](@ref): figure plotting the how the objective function varies when marginally moving the estimated parameter in one dimension at a time. If the optimum is a minimum, then the graphs should show that the objective function is minimized at that point.

* [`fbootstrap`](@ref): figure showing the histogram of parameter estimates obtained in each bootstrap iteration. 

* [`plotmoms`](@ref): figure comparing the moments in the data and in the model (at the estimated parameter values). 

* [`tableest`](@ref): table with parameter estimates, optionally including bootstrap SE.

* [`tablemoms`](@ref): table comparing the moments in the data and in the model (at the estimated parameter values), optionally including bootstrap SE.

## Model-specific files
This section describes the general format for model-specific files so that they can be used by the files performing the estimation routine. 

In general, the framework is flexible to accomodate any user-defined model. However, the following model-specific files should be included (as they are called in the general estimation routine):

* [`EstimationMode`](@ref): type for model to be estimated.
* [`InnerAuxStructure`](@ref): type for the auxiliary structures to be initialized.
* [`defaultInnerAuxStructure`](@ref): sets up default auxiliary object for computing model moments.
* [`default_weight_matrix`](@ref): sets up default weighting matrix to compute the objective function.
* [`datamoments`](@ref): computes moments from the data.
* [`parambounds`](@ref): return parameter labels and ranges.
* [`obj_mom`](@ref): computes moments in model under a given parameter guess.
* [`mdiff`](@ref): computes deviation of model moments from data moments.
* [`momentnames`](@ref): returns the full names of the moments, used for organising results.
* [`mean_moments`](@ref): computes mean of moments, used for bootstrapping.

In addition, the following can also be model-specific:
* [`indexvector`](@ref): if the default of having all model parameters estimated should be changed.
* [`ftypemom`](@ref): if not having model-specific default moments to be estimated should be changed.
* [`plotmoms`](@ref): if the default of having an empty graph to compare moments in the model and data should be changed.

### To do (not very urgent)
- avoid duplicating in documenting (e.g. `estimation` or `estimation_name`)
- default_weighting_matrix input is not ideal, maybe should have a default
- empty functions should throw error is no method is defined
- too much documentation atm, makes files hard to read. should perhaps not document everything in code, only stuff which is not obvious / which is directly used by user
- make filename_suffix more consistent
- for `moment_names` do we need two methods?

### Other comments

__Showing Progress:__ Currently we rely on https://github.com/timholy/ProgressMeter.jl. There seem to be some bugs related to show a messy output when combined with multithreading (maybe ok in mac?), e.g.
- https://github.com/timholy/ProgressMeter.jl/issues/231
- https://github.com/timholy/ProgressMeter.jl/issues/71
- https://github.com/timholy/ProgressMeter.jl/issues/151

before registering, we should check the status of these issues.




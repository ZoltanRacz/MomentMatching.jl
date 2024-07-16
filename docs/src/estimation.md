# Estimation

## Background
Generalized Method of Moments (GMM) estimation[^1] consists in finding the vector of parameters ``\theta`` that solves the following system of equations:

```math
\mathbb{E}\left[g\left(\theta\right)\right]=0
```

where ``g`` is a vector of functions representing moment conditions derived from a model depending on the vector of parameters to be estimated ``\theta``.[^2]

Letting ``k`` be the dimensionality of ``\theta`` and ``m`` that of ``g``, if:

1. ``m<k``, then the system does not have a solution;
2. ``m=k``, then the system is *just-identified* and there is a unique solution;
3. ``m>k``, then the system is *over-identified* and more than one solution is possible.

In the just-identified case, common procedures to solve system of equations can be used. However, when the system is over-identified, this approach is not possible. Instead, the goal becomes finding the vector of parameters ``\theta`` that minimizes the following objective function:

```math
g(\theta)^{\prime} W g(\theta)
```

where ``W`` is a weighting matrix.[^3] 

This package provides a series of numerical routines to find such vector ``\theta``. It also contains ready-to-use tools to check the quality of the estimation results and to do statistical inference.

## Model setup

One of the main challenges to 

```@docs
EstimationSetup
```

Here describe the different structures/elements that need to be set up before the numerical routines can be run. Highlight that these structures are flexible and general enough to accomodate estimation of any model.

## Numerical routines

### Global stage

### Local stage

### Comparison with other algorithms

## Multithreading and multiprocessing

### Locally

### Cluster

## Two-step estimation



[^1]: Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. Econometrica, 50(4), 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775).

[^2]: When moment conditions are computed via simulation, the above procedure is called Simulated Method of Moments (SMM) estimation. For more details: McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. Econometrica, 57(5), 995–1026. [https://doi.org/10.2307/1913621](https://doi.org/10.2307/1913621).

[^3]: Clearly, this procedure can also be used in the just-identified case and it should deliver the same result as the one obtained when using routines to solve system of equations.
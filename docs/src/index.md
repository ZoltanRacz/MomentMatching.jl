# MomentMatching.jl 

This package provides a unified framework to perform Method of Moments (Generalized and Simulated) parameter estimation for economic models.

Generalized Method of Moments (GMM) estimation[^1] consists in finding the vector of parameters ``\theta`` that solves the following system of equations:

```math
\mathbb{E}\left[g\left(\theta\right)\right]=0
```

where ``g`` is a vector of functions representing moment conditions derived from a model depending on the vector of parameters to be estimated ``\theta``. When moment conditions are computed via simulation, the above procedure is called Simulated Method of Moments (SMM) estimation.[^2]

When the system is over-identified (``g`` has higher dimensionality than ``\theta``), solving such an equation exactly is in general not possible. Instead, the goal becomes finding the vector of parameters ``\theta`` that minimizes the following objective function:

```math
g(\theta)^{\prime} W g(\theta)
```

where ``W`` is a weighting matrix.[^3] 

## Features

This package aims at providing a general toolbox for such estimation exercises, irrespective of what model one wants to estimate. Ready-to-use tools are provided to:
1. estimate ``\theta`` via a combination of global search and a local optimization routine;
2. perform diagnostic checks on the results; 
3. do statistical inference and 
4. produce tables and figures displaying the estimation results.

The package was built having in mind three key features:
- *Ease of use*: The user just needs to write a wrapper around her code conformable with the estimation routines and set the algorithm options before running the main estimation function. This is an easy task even if the model code was not written specifically to be compatible with this package. 
- *Flexibility*: Routines can be used for the estimation of any model. Estimation of robustness checks and alternative model specifications is convenient.
- *Parallelization*: Multithreading and multiprocessing (also combined) are supported, both locally and on a cluster.

## Installation
To install the package run:
```
using Pkg
Pkg.add("MomentMatching")
```
To load the package use the command:
```
using MomentMatching
```

## Authors and citing
- [Gualtiero Azzalini](https://gualtiazza.github.io/) (Stockholm School of Economics) 
- [Zoltán Rácz](https://www.zoltanracz.net/) (Stockholm School of Economics)

Please consider citing the package if you use MomentMatching in your research:
```
@misc{MomentMatching.jl,
  title = {{MomentMatching.jl}: Parameter Estimation of Economic Models via Moment Matching Methods in Julia},
  author = {Azzalini, Gualtiero and Rácz, Zoltán},
  year = {2024},
  howpublished = {\url{https://github.com/ZoltanRacz/MomentMatching.jl}}
}
```

[^1]: Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. Econometrica, 50(4), 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775).

[^2]: McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. Econometrica, 57(5), 995–1026. [https://doi.org/10.2307/1913621](https://doi.org/10.2307/1913621).

[^3]: Clearly, this procedure can also be used in the just-identified case and it should deliver the same result as the one obtained when using routines to solve system of equations.

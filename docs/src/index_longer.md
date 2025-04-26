# MomentMatching.jl 

Generalized Method of Moments (GMM) estimation[^1] consists in finding the vector of parameters ``\theta`` that solves the following system of equations:

```math
\mathbb{E}\left[g\left(\theta\right)\right]=0
```

where ``g`` is a vector of functions representing moment conditions derived from a model depending on the vector of parameters to be estimated ``\theta``. When moment conditions are computed via simulation, the above procedure is called Simulated Method of Moments (SMM) estimation.[^2]

Letting ``k`` be the dimensionality of ``\theta`` and ``m`` that of ``g``, if:

1. ``m<k``, then the system does not have a solution;
2. ``m=k``, then the system is *just-identified* and there is a unique solution;
3. ``m>k``, then the system is *over-identified* and more than one solution is possible.

In the just-identified case, common procedures to solve system of equations can be used. However, when the system is over-identified, this approach is not possible. Instead, the goal becomes finding the vector of parameters ``\theta`` that minimizes the following objective function:

```math
g(\theta)^{\prime} W g(\theta)
```

where ``W`` is a weighting matrix.[^3] 

This package provides a series of numerical routines to perform Method of Moments estimation (Generalized and Simulated), i.e., find an estimate of the vector ``\theta`` (see section [Estimation](@ref)). It also contains ready-to-use tools (i) to check the quality of the estimation results and to do statistical inference (see section [Inference and Diagnostics](@ref)) and (ii) to produce tables and figures displaying estimation results (see section [Output](@ref)).

The package was built having in mind three key features:
- *Flexibility*: routines can be used for estimation of any model;
- *Parallelization*: multithreading and multiprocessing (also combined) are supported, both locally and on a cluster;
- *Ease of use*: the user just needs to write her model in a way conformable with the estimation routines and set the algorithm options before running the main estimation function.


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

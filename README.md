# MomentMatching

[![Build Status](https://github.com/ZoltanRacz/MomentMatching.jl/actions/workflows/CI.yml/badge.svg?event=push)](https://github.com/ZoltanRacz/MomentMatching.jl/actions/workflows/CI.yml?event=push)
[![codecov](https://codecov.io/gh/ZoltanRacz/MomentMatching.jl/graph/badge.svg?token=YLP96BUQ9S)](https://codecov.io/gh/ZoltanRacz/MomentMatching.jl)

by Gualtiero Azzalini (Stockholm School of Economics)\
Zoltan Racz (Stockholm School of Economics)


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

## Documentation

The online documentation is available at ...

!!! note
    A breaking change with cleaner syntax and more detailed documentation can be expected in 2025.

A worked out example with the current version of the code is shown at ...
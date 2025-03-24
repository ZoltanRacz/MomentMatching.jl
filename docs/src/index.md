# MomentMatching.jl 

This is a package to perform Method of Moments estimation (Generalized and Simulated). It was built having in mind three key features:
- *Flexibility*: routines can be used for estimation of any model;
- *Parallelization*: multithreading and multiprocessing (also combined) are supported, both locally and on a cluster;
- *Ease of use*: the user just needs to write her model in the way described later and set the algorithm options before running the main estimation function.

The package consists of three main files:
1. *estimation.jl*: contains structures and functions to perform estimation (see section [Estimation](@ref));
2. *inference.jl*: contains functions to perform inference and diagnostics tests (see section [Inference and Diagnostics](@ref));
3. *output.jl*: contains structures and functions to produce tables and figures displaying estimation results (see section [Output](@ref)).

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
Authors:
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

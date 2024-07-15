# MomentMatching.jl 

This is a package to perform Method of Moments estimation (Generalized and Simulated). It was built having in mind three key features:
- *Flexibility*: routines can be used for estimation of any model;
- *Parallelization*: multithreading and multiprocessing (also combined) are supported, both locally and on a cluster;
- *Ease of use*: the user just needs to write her model in the way described later and set the algorithm options before running the main estimation function.

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
## Authors

- Gualtiero Azzalini (Stockholm School of Economics) 
- Zoltán Rácz (Stockholm School of Economics)


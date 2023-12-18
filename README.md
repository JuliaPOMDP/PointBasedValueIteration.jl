# Point-based value iteration

[![CI](https://github.com/JuliaPOMDP/PointBasedValueIteration.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/PointBasedValueIteration.jl/actions/workflows/CI.yml)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/PointBasedValueIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/PointBasedValueIteration.jl?branch=master)


Point-based value iteration solver ([Pineau et al., 2003](http://www.fore.robot.cc/papers/Pineau03a.pdf), [Shani et al., 2012](https://link.springer.com/content/pdf/10.1007/s10458-012-9200-2.pdf)) for the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

## Installation
This package is available from Julia's General package registry.
```
using Pkg
Pkg.add("PointBasedValueIteration")
```

## Usage
```
using PointBasedValueIteration
using POMDPModels
pomdp = TigerPOMDP() # initialize POMDP

solver = PBVISolver() # set the solver

policy = solve(solver, pomdp) # solve the POMDP
```

The function `solve` returns an `AlphaVectorPolicy` as defined in [POMDPTools](https://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/policies/).

## References
- Pineau, J., Gordon, G., & Thrun, S. (2003, August). Point-based value iteration: An anytime algorithm for POMDPs. In IJCAI (Vol. 3, pp. 1025-1032).
- Shani, G., Pineau, J. & Kaplow, R. A survey of point-based POMDP solvers. Auton Agent Multi-Agent Syst 27, 1–51 (2013). https://doi.org/10.1007/s10458-012-9200-2

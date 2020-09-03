# Point-based value iteration

[![Build status](https://travis-ci.com/JuliaPOMDP/PointBasedValueIteration.jl.svg?branch=master)](https://travis-ci.com/github/JuliaPOMDP/PointBasedValueIteration.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/PointBasedValueIteration.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/PointBasedValueIteration.jl?branch=master)
[![codecov](https://codecov.io/gh/JuliaPOMDP/PointBasedValueIteration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/PointBasedValueIteration.jl)


Point-based value iteration solver ([Pineau et al., 2003](http://www.fore.robot.cc/papers/Pineau03a.pdf)) for the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

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

solver = PBVI(n_belief_points=100, max_iterations=100) # set the solver

policy = solve(solver, pomdp) # solve the POMDP
```

The function `solve` returns an `AlphaVectorPolicy` as defined in [POMDPPolicies](https://github.com/JuliaPOMDP/POMDPPolicies.jl).

## References
- Pineau, J., Gordon, G., & Thrun, S. (2003, August). Point-based value iteration: An anytime algorithm for POMDPs. In IJCAI (Vol. 3, pp. 1025-1032).

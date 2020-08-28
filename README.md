# Point-based value iteration

[![Build status](https://travis-ci.com/dominikstrb/PointBasedValueIteration.jl.svg?branch=master)](https://travis-ci.com/github/dominikstrb/PointBasedValueIteration.jl)

Point-based value iteration solver ([Pineau et al., 2003](http://www.fore.robot.cc/papers/Pineau03a.pdf)) for the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

## Installation

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

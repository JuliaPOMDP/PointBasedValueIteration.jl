# Point-based value iteration

Point-based value iteration solver ([Pineau et al., 2003](http://www.fore.robot.cc/papers/Pineau03a.pdf)) for the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

## TODO:
- [ ] compare the alpha vectors to the output of SARSOP or IncrementalPruning by running `value(policy, b)` for many random choices of b for both a policy from SARSOP/IncrementalPruning and one from PBVI on several different problems
- [ ] Documentation (example) in the README

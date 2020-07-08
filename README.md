# Point-based value iteration

Code adapted from https://github.com/kylewray/julia_pomdp/blob/4d1724b7b15e999eff68f9b3bb632ace89cdc4e1/src/pomdp/offline/PBVI.jl

## TODO:
- [ ] compare the alpha vectors to the output of SARSOP or IncrementalPruning by running `value(policy, b)` for many random choices of b for both a policy from SARSOP/IncrementalPruning and one from PBVI on several different problems

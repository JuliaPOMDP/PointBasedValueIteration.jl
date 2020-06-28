using POMDPModels
using POMDPSimulators
using POMDPs

include("../src/PointBasedValueIteration.jl")
using .PointBasedValueIteration

pomdp = TigerPOMDP()

solver = PBVI(100, 100)
policy = solve(solver, pomdp)

sim = RolloutSimulator(max_steps=100)

rs_pbvi = [simulate(sim, pomdp, policy) for _ in 1:1000]

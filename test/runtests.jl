using POMDPModels
using POMDPSimulators
using POMDPs

include("../src/PointBasedValueIteration.jl")
using .PointBasedValueIteration

pomdp = TigerPOMDP()

solver = PBVI(n_belief_points=100, max_iterations=100)
policy = solve(solver, pomdp)

sim = RolloutSimulator(max_steps=100)

rs_pbvi = [simulate(sim, pomdp, policy) for _ in 1:1000]

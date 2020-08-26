using Test
using POMDPModels
using POMDPSimulators
using POMDPs
using SARSOP
using BeliefUpdaters

using PointBasedValueIteration

@testset "Comparison with SARSOP" begin
    pomdps = [TigerPOMDP(), BabyPOMDP()]

    for pomdp in pomdps
        solver = PBVI(n_belief_points=200, max_iterations=500)
        policy = solve(solver, pomdp)

        sarsop = SARSOPSolver()
        sarsop_policy = solve(sarsop, pomdp)

        B = [DiscreteBelief(pomdp, [b, 1-b]) for b in 0:0.01:1]

        pbvi_vals = [value(policy, b) for b in B]
        sarsop_vals = [value(sarsop_policy, b) for b in B]

        @test isapprox(sarsop_vals, pbvi_vals, rtol=0.05)
    end

end

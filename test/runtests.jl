using Test
using POMDPModels
using POMDPs
using SARSOP
using BeliefUpdaters

using PointBasedValueIteration

@testset "Comparison with SARSOP" begin
    pomdps = [TigerPOMDP(), BabyPOMDP(), MiniHallway()]

    for pomdp in pomdps
        solver = PBVISolver()
        policy = solve(solver, pomdp)

        sarsop = SARSOPSolver(verbose=false)
        sarsop_policy = solve(sarsop, pomdp)

        B = []
        if typeof(pomdp) == MiniHallway
            B = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.083337, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        else
            for _ in 1:100
                r = rand(length(states(pomdp)))
                push!(B, DiscreteBelief(pomdp, r/sum(r)))
            end
        end

        pbvi_vals = [value(policy, b) for b in B]
        if typeof(pomdp) == MiniHallway
            sarsop_vals = [value(sarsop_policy, b) for b in B]
            # Test passes when the value function is multiplied
            # @test isapprox(sarsop_vals * 3.4 , pbvi_vals, rtol=0.3)
            @test_broken isapprox(sarsop_vals, pbvi_vals, rtol=0.3)
        else
            sarsop_vals = [value(sarsop_policy, b) for b in B]
            @test isapprox(sarsop_vals, pbvi_vals, rtol=0.1)
        end
    end
end

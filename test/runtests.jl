using Test
using POMDPModels
using POMDPs
using SARSOP
using BeliefUpdaters
using POMDPModelTools: Deterministic
using POMDPSimulators: RolloutSimulator

using PointBasedValueIteration

@testset "Comparison with SARSOP" begin
    pomdps = [TigerPOMDP(), BabyPOMDP(), MiniHallway()]

    for pomdp in pomdps
        solver = PBVISolver(10, typeof(pomdp) == MiniHallway ? 0.05 : 0.01, true)
        policy = solve(solver, pomdp)

        sarsop = SARSOPSolver(verbose=true)
        sarsop_policy = solve(sarsop, pomdp)

        @testset "$(typeof(pomdp)) Value function comparison" begin
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
            sarsop_vals = [value(sarsop_policy, b) for b in B]
            @test isapprox(sarsop_vals, pbvi_vals, rtol=0.1)
        end

        @testset "$(typeof(pomdp)) Simulation results comparison" begin
            no_simulations = typeof(pomdp) == MiniHallway ? 1 : 10_000
            for s in states(pomdp)
                # println(s)
                # @show value(policy, Deterministic(s))
                # @show value(sarsop_policy, Deterministic(s))
                #
                # @show action(policy, Deterministic(s))
                # @show action(sarsop_policy, Deterministic(s))
                #
                # @show mean([simulate(RolloutSimulator(max_steps = 100), pomdp, policy, updater(policy), Deterministic(s)) for i in 1:no_simulations])
                # @show mean([simulate(RolloutSimulator(max_steps = 100), pomdp, sarsop_policy, updater(sarsop_policy), Deterministic(s)) for i in 1:no_simulations])

                # In this state the PBVI outputs better results than SARSOP, because SARSOP does not evaluate this state, thus having sub-optimal result
                if s == 5 && typeof(pomdp) == MiniHallway
                    @test_broken isapprox(value(policy, Deterministic(s)), value(sarsop_policy, Deterministic(s)), rtol=0.1)
                    @test_broken isapprox(  mean([simulate(RolloutSimulator(max_steps = 100), pomdp, policy, updater(policy), Deterministic(s)) for i in 1:no_simulations]),
                                            mean([simulate(RolloutSimulator(max_steps = 100), pomdp, sarsop_policy, updater(sarsop_policy), Deterministic(s)) for i in 1:no_simulations]),
                                            rtol=0.1)
                else
                    @test isapprox(value(policy, Deterministic(s)), value(sarsop_policy, Deterministic(s)), rtol=0.1)
                    @test isapprox( mean([simulate(RolloutSimulator(max_steps = 100), pomdp, policy, updater(policy), Deterministic(s)) for i in 1:no_simulations]),
                                    mean([simulate(RolloutSimulator(max_steps = 100), pomdp, sarsop_policy, updater(sarsop_policy), Deterministic(s)) for i in 1:no_simulations]),
                                    rtol=0.1)
                end
            end
        end
    end
end

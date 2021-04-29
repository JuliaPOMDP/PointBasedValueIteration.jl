using Test
using POMDPModels
using POMDPs
using SARSOP
using BeliefUpdaters
using POMDPModelTools: Deterministic
using POMDPSimulators: RolloutSimulator
using FiniteHorizonPOMDPs

using PointBasedValueIteration

@testset "Convert test" begin
    @testset "Infinite Horizon POMDP tests" begin
        tigerPOMDP = TigerPOMDP()
        babyPOMDP = BabyPOMDP()
        minihallwayPOMDP = MiniHallway()

        @test convert(Array{Float64, 1}, initialstate(tigerPOMDP), tigerPOMDP) == [0.5, 0.5]
        @test convert(Array{Float64, 1}, initialstate(babyPOMDP), babyPOMDP) == [1., 0.]
        @test convert(Array{Float64, 1}, initialstate(minihallwayPOMDP), minihallwayPOMDP) == append!(fill(1/12, 12), zeros(1))
    end

    @testset "Finite Horizon POMDP tests" begin
        @testset "Finite Horizon POMDP initial state convert tests" begin
            tigerPOMDP = fixhorizon(TigerPOMDP(), 1)
            babyPOMDP = fixhorizon(BabyPOMDP(), 1)
            minihallwayPOMDP = fixhorizon(MiniHallway(), 1)

            @test convert(Array{Float64, 1}, initialstate(tigerPOMDP), tigerPOMDP) == [0.5, 0.5, 0., 0.]
            @test convert(Array{Float64, 1}, initialstate(babyPOMDP), babyPOMDP) == [1., 0., 0., 0.]
            @test convert(Array{Float64, 1}, initialstate(minihallwayPOMDP), minihallwayPOMDP) == append!(fill(1/12, 12), zeros(14))
        end

        @testset "Finite Horizon POMDP other than initial stage distribution tests" begin
            tigerPOMDP = fixhorizon(TigerPOMDP(), 2)
            babyPOMDP = fixhorizon(BabyPOMDP(), 2)
            minihallwayPOMDP = fixhorizon(MiniHallway(), 2)

            tigerbelief = FiniteHorizonPOMDPs.InStageDistribution(FiniteHorizonPOMDPs.distribution(initialstate(tigerPOMDP)), 2)
            babybelief = FiniteHorizonPOMDPs.InStageDistribution(FiniteHorizonPOMDPs.distribution(initialstate(babyPOMDP)), 2)
            minihallwaybelief = FiniteHorizonPOMDPs.InStageDistribution(FiniteHorizonPOMDPs.distribution(initialstate(minihallwayPOMDP)), 2)

            @test convert(Array{Float64, 1}, tigerbelief, tigerPOMDP) == [0., 0., 0.5, 0.5, 0., 0.]
            @test convert(Array{Float64, 1}, babybelief, babyPOMDP) == [0., 0., 1., 0., 0., 0.]
            @test convert(Array{Float64, 1}, minihallwaybelief, minihallwayPOMDP) == append!(append!(zeros(13), fill(1/12, 12)), zeros(14))
        end
    end
end

@testset "Comparison with SARSOP" begin
    pomdps = [TigerPOMDP(), BabyPOMDP(), MiniHallway()]

    for pomdp in pomdps
        solver = PBVISolver(10, typeof(pomdp) == MiniHallway ? 0.05 : 0.01, false)
        policy = solve(solver, pomdp)

        sarsop = SARSOPSolver(verbose=false)
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
                    continue
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

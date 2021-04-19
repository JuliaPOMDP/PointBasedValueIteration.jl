"""
    PBVISolver <: Solver
POMDP solver type using point-based value iteration
"""
mutable struct PBVISolver <: Solver
    max_iterations::Int64
    ϵ::Float64
    verbose::Bool
end

"""
    PBVISolver(; max_iterations, tolerance)
Initialize a point-based value iteration solver with default `max_iterations` and ϵ.
"""
function PBVISolver(;max_iterations::Int64=10, ϵ::Float64=0.01, verbose::Bool=false)
    return PBVISolver(max_iterations, ϵ, verbose)
end

"""
    AlphaVec
Alpha vector type of paired vector and action.
"""
struct AlphaVec
    alpha::Vector{Float64} # alpha vector
    action::Any # action associated wtih alpha vector
end

# define alpha vector equality
==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))

convert(::Type{Array{Float64, 1}}, d::BoolDistribution, pomdp) = [d.p, 1 - d.p]
convert(::Type{Array{Float64, 1}}, d::DiscreteUniform, pomdp) = [pdf(d, stateindex(pomdp, s)) for s in states(pomdp)]
convert(::Type{Array{Float64, 1}}, d::SparseCat, pomdp) = d.probs

convert(::Type{Array{Float64, 1}}, d::InStageDistribution{DiscreteUniform}, m::FixedHorizonPOMDPWrapper) = vec([pdf(d, s) for s in states(m)])
convert(::Type{Array{Float64, 1}}, d::InStageDistribution{BoolDistribution}, m::FixedHorizonPOMDPWrapper) = [[d.d.p[1], 1 - d.d.p[1]]..., zeros(length(states(m)) - 2)...]


function _argmax(f, X)
    return X[argmax(map(f, X))]
end

function belief_update(pomdp, b, b′, terminals, not_terminals)
    if sum(b′[not_terminals]) != 0.
        if !isempty(terminals)
            b′[not_terminals] = b′[not_terminals] / (sum(b′[not_terminals]) / (1. - sum(b[terminals]) - sum(b′[terminals])))
            b′[terminals] += b[terminals]
        else
            b′[not_terminals] /= sum(b′[not_terminals])
        end
    else
        b′[terminals] += b[terminals]
        b′[terminals] /= sum(b′[terminals])
    end
    return b′
end

function backup_belief(pomdp::POMDP, Γ, b)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    Γa = Vector{Vector{Float64}}(undef, length(A))

    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    for a in A
        Γao = Vector{Vector{Float64}}(undef, length(O))
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

        for o in O
            # update beliefs
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs

            if sum(b′) > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, belief_update(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end

            # extract optimal alpha vector at resulting belief
            Γao[obsindex(pomdp, o)] = _argmax(α -> α ⋅ b′.b, Γ)
        end

        # construct new alpha vectors
        Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                        for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                        for s in S]
    end

    # find the optimal alpha vector
    idx = argmax(map(αa -> αa ⋅ b.b, Γa))
    alphavec = AlphaVec(Γa[idx], A[idx])

    return alphavec
end


function improve(pomdp, B, Γ, solver)
    alphavecs = nothing
    while true
        Γold = Γ
        alphavecs = [backup_belief(pomdp, Γold, b) for b in B]
        Γ = [alphavec.alpha for alphavec in alphavecs]
        prec = max([sum(abs.(α1 .- α2)) for (α1, α2) in zip(Γold, Γ)]...)
        if solver.verbose println("    Improving alphas, maximum gap between old and new α vector: $(prec)") end
        prec > solver.ϵ || break
    end

    return Γ, alphavecs
end

function successors(pomdp, b, Bs)
    S = ordered_states(pomdp)
    not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
    terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
    succs = []

    for a in actions(pomdp)
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b[terminals] end

        for o in observations(pomdp)
            #update belief
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs


            if sum(b′) > 0.
                b′ = belief_update(pomdp, b, b′, terminals, not_terminals)

                if !in(b′, Bs)
                    push!(succs, b′)
                end
            end
        end
    end

    return succs
end

function succ_dist(pomdp, bp, B)
    dist = [norm(bp - b.b, 1) for b in B]
    return max(dist...)
end

function expand(pomdp, B, Bs)
    B_new = copy(B)
    for b in B
        succs = successors(pomdp, b.b, Bs)
        if length(succs) > 0
            b′ = succs[argmax([succ_dist(pomdp, bp, B) for bp in succs])]
            push!(B_new, DiscreteBelief(pomdp, b′))
            push!(Bs, b′)
        end
    end

    return B_new, Bs
end

# 1: B ← {b0}
# 2: while V has not converged to V∗ do
# 3:    Improve(V, B)
# 4:    B ← Expand(B)
function solve(solver::PBVISolver, pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    # best action worst state lower bound
    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    #init belief, if given distribution, convert to vector
    init = convert(Array{Float64, 1}, initialstate(pomdp), pomdp)
    B = [DiscreteBelief(pomdp, init)]
    Bs = Set([init])

    if solver.verbose println("Running PBVI solver on $(typeof(pomdp)) problem with following settings:\n    max_iterations = $(solver.max_iterations), ϵ = $(solver.ϵ), verbose = $(solver.verbose)\n+----------------------------------------------------------+") end

    # original code should run until V converges to V*, this yet needs to be implemented
    # for example as: while max(@. abs(newV - oldV)...) > solver.ϵ
    # However this probably would not work, as newV and oldV have different number of elements (arrays of alphas)
    alphavecs = nothing
    for i in 1:solver.max_iterations
        Γ, alphavecs = improve(pomdp, B, Γ, solver)
        B, Bs = expand(pomdp, B, Bs)
        if solver.verbose println("Iteration $(i) executed, belief set contains $(length(Bs)) belief vectors.") end
    end

    acts = [alphavec.action for alphavec in alphavecs]
    return AlphaVectorPolicy(pomdp, Γ, acts)
end


@POMDPLinter.POMDP_require solve(solver::PBVISolver, pomdp::POMDP) begin
    P = typeof(pomdp)
    S = state_type(P)
    A = action_type(P)
    O = observation_type(P)
    @req discount(::P) # discount factor
    @subreq ordered_states(pomdp)
    @subreq ordered_actions(pomdp)
    @subreq ordered_observations(pomdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A)
    ss = states(pomdp)
    as = actions(pomdp)
    os = observations(pomdp)
    @req length(::typeof(ss))
    s = first(iterator(ss))
    a = first(iterator(as))
    dist = transition(pomdp, s, a)
    D = typeof(dist)
    @req pdf(::D,::S)

    odist = observation(pomdp, a, s)
    OD = typeof(odist)
    @req pdf(::OD,::O)
end

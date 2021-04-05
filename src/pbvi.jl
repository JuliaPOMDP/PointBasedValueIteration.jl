"""
    PBVISolver <: Solver
POMDP solver type using point-based value iteration
"""
mutable struct PBVISolver <: Solver
    max_iterations::Int64
    ϵ::Float64
end

"""
    PBVISolver(; max_iterations, tolerance)
Initialize a point-based value iteration solver with default `max_iterations` and ϵ.
"""
function PBVISolver(;max_iterations::Int64=10, ϵ::Float64=0.01)
    return PBVISolver(max_iterations, ϵ)
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


function _argmax(f, X)
    return X[argmax(map(f, X))]
end

function backup_belief(pomdp::POMDP, Γ, b)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    Γa = Vector{Vector{Float64}}(undef, length(A))

    for a in A
        Γao = Vector{Vector{Float64}}(undef, length(O))
        trans_probs = sum([pdf(transition(pomdp, s, a), sp) * b.b[stateindex(pomdp, s)] for s in S, sp in S], dims=1)

        for o in O
            # update beliefs
            b′ = nothing
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), ordered_states(pomdp)), o)
            pr_o_given_b_a = sum(obs_probs .* vec(trans_probs))

            # P(o|b, a) = ∑(sp∈S) P(o|a, sp) ∑(s∈S) P(sp|s, a) * b(s)
            if pr_o_given_b_a > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, [obs_probs[stateindex(pomdp, sp)] / pr_o_given_b_a * trans_probs[stateindex(pomdp, sp)] for sp in ordered_states(pomdp)])
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end
            # extract optimal alpha vector at resulting belief
            Γao[obsindex(pomdp, o)] = _argmax(α -> α ⋅ b′.b, Γ)
        end

        # construct new alpha vectors
        αa = [r(s, a) + γ * sum(sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                  for (j, sp) in enumerate(S))
                              for (i, o) in enumerate(O))
              for s in S]

        Γa[actionindex(pomdp, a)] = αa
    end

    # find the optimal alpha vector
    idx = argmax(map(αa -> αa ⋅ b.b, Γa))
    alphavec = AlphaVec(Γa[idx], A[idx])

    return alphavec
end


function improve(pomdp, B, Γ, ϵ)
    alphavecs = nothing
    while true
        Γold = Γ
        alphavecs = [backup_belief(pomdp, Γold, b) for b in B]
        Γ = [alphavec.alpha for alphavec in alphavecs]
        max([sum(abs.(α1 .- α2)) for (α1, α2) in zip(Γold, Γ)]...) .> ϵ || break
    end

    return Γ, alphavecs
end

function successors(pomdp, b, Bs)
    succs = []
    for a in actions(pomdp)
        trans_probs = sum([pdf(transition(pomdp, s, a), sp) * b[stateindex(pomdp, s)] for s in states(pomdp), sp in ordered_states(pomdp)], dims=1)
        for o in observations(pomdp)
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), ordered_states(pomdp)), o)
            pr_o_given_b_a = sum(obs_probs .* trans_probs')
            if pr_o_given_b_a > 0.
                b′ = [obs_probs[stateindex(pomdp, sp)] / pr_o_given_b_a * trans_probs[stateindex(pomdp, sp)] for sp in ordered_states(pomdp)]
                if !in(b′, Bs)
                    push!(succs, b′)
                end
            end

            # This should work but does not, update throws error every time

            # try
            #     b′ = update(DiscreteUpdater(pomdp), b, a, o)
            #     if !in(b′, Bs)
            #         push!(succs, b′)
            #     end
            # catch e
            #     nothing
            # end
        end
    end

    return succs
end

function succ_dist(pomdp, bp, B)
    dist = [norm(bp - b.b, 1) for b in B]
    return max(dist...)
end

function expand(pomdp, B, Bs)
    B_new = deepcopy(B)
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

    #init belief, if given Distribution, convert to vector
    init = initialstate(pomdp)
    if typeof(init) <: BoolDistribution || typeof(init) <: DiscreteUniform
        init = convert(Array{Float64, 1}, init, pomdp)
    end
    B = [DiscreteBelief(pomdp, init)]
    Bs = Set([init])

    # original code should run until V converges to V*, this yet needs to be implemented
    # for example as: while max(@. abs(newV - oldV)...) > solver.ϵ
    # However this probably would not work, as newV and oldV have different number of elements (arrays of alphas)
    alphavecs = nothing
    for i in 1:solver.max_iterations
        Γ, alphavecs = improve(pomdp, B, Γ, solver.ϵ)
        B, Bs = expand(pomdp, B, Bs)
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

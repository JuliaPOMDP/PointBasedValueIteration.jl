"""
    PBVI <: Solver
POMDP solver type using point-based value iteration
"""
mutable struct PBVI <: Solver
    max_iterations::Int64
    ϵ::Float64
end

"""
    PBVI(; max_iterations, tolerance)
Initialize a point-based value iteration solver with default `max_iterations` and ϵ.
"""
function PBVI(;max_iterations::Int64=10, ϵ::Float64=0.01)
    return PBVI(max_iterations, ϵ)
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

convert(::Type{Array{Float64, 1}}, d::BoolDistribution)  = [d.p, 1 - d.p]
convert(::Type{Array{Float64, 1}}, d::SparseCat)  = d.probs

# P(o|b, a) = ∑(sp∈S) P(o|a, sp) ∑(s∈S) P(sp|s, a) * b(s)
function prob_o_given_b_a(pomdp, b::Array{Float64}, a, o)
    pr_o_given_b_a = 0.
    for sp in states(pomdp)
        temp_sum = sum([pdf(transition(pomdp, s, a), sp) * b[stateindex(pomdp, s)] for s in states(pomdp)])
        pr_o_given_b_a += pdf(observation(pomdp, a, sp), o) * temp_sum
    end

    return pr_o_given_b_a
end

function b_o_a(pomdp, b::Vector{Float64}, a, o)
    b_new = zeros(length(states(pomdp)))
    for sp in states(pomdp)
        b_temp = 0
        for s in states(pomdp)
            b_temp += pdf(transition(pomdp, s, a), sp) * b[stateindex(pomdp, s)]
        end

        b_new[stateindex(pomdp, sp)] = pdf(observation(pomdp, a, sp), o) / prob_o_given_b_a(pomdp, b, a, o) * b_temp
    end

    return b_new
end

function _argmax(f, X)
    return X[argmax(map(f, X))]
end

function backup_belief(pomdp::POMDP, Γ, b)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    Γa = Vector{Float64}[]

    for a in A
        Γao = Vector{Float64}[]

        for o in O
            # update beliefs
            b′ = nothing
            try
                b′ = update(DiscreteUpdater(pomdp), b, a, o)
            catch
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end
            # extract optimal alpha vector at resulting belief
            push!(Γao, _argmax(α -> α ⋅ b′.b, Γ))
        end

        # construct new alpha vectors
        αa = [r(s, a) + γ * sum(sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                  for (j, sp) in enumerate(S))
                              for (i, o) in enumerate(O))
              for s in S]

        push!(Γa, αa)
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
        max([max(abs.(α1 .- α2)...) for (α1, α2) in zip(Γold, Γ)]...) .> ϵ || break
    end

    return Γ, alphavecs
end

function successors(pomdp, b, Bs)
    succs = []
    for a in actions(pomdp)
        for o in observations(pomdp)
            if prob_o_given_b_a(pomdp, b, a, o) > 0.
                b′ = b_o_a(pomdp, b, a, o)
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
function solve(solver::PBVI, pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)
    r = StateActionReward(pomdp)

    # best action worst state lower bound
    α_init = 1 / (1 - γ) * maximum(minimum(r(s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]

    #init belief, if given Distribution, convert to vector
    init = initialstate(pomdp)
    if typeof(init) <: BoolDistribution
        init = convert(Array{Float64, 1}, init)
    elseif typeof(init) <: SparseCat
        init = convert(Array{Float64, 1}, init)
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


@POMDPLinter.POMDP_require solve(solver::PBVI, pomdp::POMDP) begin
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

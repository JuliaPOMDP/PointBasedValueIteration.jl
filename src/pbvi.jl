"""
    PBVI <: Solver
POMDP solver type using point-based value iteration
"""
mutable struct PBVI <: Solver
    n_belief_points::Int64
    max_iterations::Int64
end

"""
    PBVI(; max_iterations, tolerance)
Initialize a point-based value iteration solver with default `n_belief_points` and `max_iterations`.
"""
function PBVI(;n_belief_points::Int64=100, max_iterations::Int64=100)
    return PBVI(n_belief_points, max_iterations)
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


function _argmax(f, X)
    return X[argmax(map(f, X))]
end

function backup_belief(pomdp::POMDP, Γ, b)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)

    Γa = Vector{Float64}[]

    for a in A
        Γao = Vector{Float64}[]

        for o in O
            b′ = update(DiscreteUpdater(pomdp), b, a, o)
            push!(Γao, _argmax(α -> α ⋅ b′.b, Γ))
        end

        αa = [reward(pomdp, s, a) + γ * sum(sum(pdf(transition(pomdp, s, a), s′) * pdf(observation(pomdp, a, s′), o) * Γao[i][j]
                                  for (j, s′) in enumerate(S))
                              for (i, o) in enumerate(O))
              for s in S]

        push!(Γa, αa)
    end

    idx = argmax(map(αa -> αa ⋅ b.b, Γa))
    alphavec = AlphaVec(Γa[idx], A[idx])

    # return _argmax(αa -> αa ⋅ b.b, Γa)
    return alphavec
end

function solve(solver::PBVI, pomdp::POMDP)
    k_max = solver.max_iterations
    s = 1 / solver.n_belief_points
    B = [DiscreteBelief(pomdp, [b, 1-b]) for b in 0:s:1]
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    γ = discount(pomdp)

    α_init = 1 / (1 - γ) * maximum(minimum(reward(pomdp, s, a) for s in S) for a in A)
    Γ = [fill(α_init, length(S)) for a in A]
    res = nothing

    for k in 1 : k_max
        res = [backup_belief(pomdp, Γ, b) for b in B]
        Γ = [alphavec.alpha for alphavec in res]
    end

    acts = [alphavec.action for alphavec in res]
    return AlphaVectorPolicy(pomdp, Γ, acts)
end


@POMDP_require solve(solver::PBVI, pomdp::POMDP) begin
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
    @req iterator(::typeof(as))
    @req iterator(::typeof(ss))
    @req iterator(::typeof(os))
    s = first(iterator(ss))
    a = first(iterator(as))
    dist = transition(pomdp, s, a)
    D = typeof(dist)
    @req pdf(::D,::S)

    odist = observation(pomdp, a, s)
    OD = typeof(odist)
    @req pdf(::OD,::O)
end

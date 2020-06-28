module PointBasedValueIteration

using POMDPs
using POMDPPolicies
using POMDPModelTools
using BeliefUpdaters
using LinearAlgebra

import POMDPs: Solver
import Base: ==, hash


export
    PBVI,
    solve

include("pbvi.jl")

end # module

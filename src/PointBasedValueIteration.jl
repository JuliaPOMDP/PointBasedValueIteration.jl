module PointBasedValueIteration

using POMDPs
using POMDPPolicies
using POMDPModelTools
using POMDPLinter
using BeliefUpdaters
using LinearAlgebra

import POMDPs: Solver, solve
import Base: ==, hash


export
    PBVI,
    solve

include("pbvi.jl")

end # module

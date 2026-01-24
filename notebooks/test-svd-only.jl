using EnsembleKalmanProcesses
using LinearAlgebra, Statistics, Random

Random.seed!(0) 

output_dim = parse(Int, ARGS[1])
n_trials   = parse(Int, ARGS[2])

Y = randn(output_dim, n_trials)
Y = Y .- mean(Y, dims = 2)

internal_cov = tsvd_cov_from_samples(Y)

C_ref = Y * Y' / (n_trials - 1)

error = Matrix(internal_cov) - C_ref

println("Error (relative): ", norm(error) / norm(C_ref))

using Pkg
println("Julia version: ", VERSION)
Pkg.status("EnsembleKalmanProcesses")
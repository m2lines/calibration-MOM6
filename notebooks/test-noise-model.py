from julia import Main

import numpy as np

noise_ens = np.random.randn(1000, 100)
ens_size = noise_ens.shape[1]

noise_ens = noise_ens - noise_ens.mean(1,keepdims=True)

Main.noise_ens = noise_ens

Main.eval("""
using EnsembleKalmanProcesses, Random     
using LinearAlgebra
internal_cov = tsvd_cov_from_samples(noise_ens)
""")

U, s, Vt = np.linalg.svd(noise_ens, full_matrices=False)

# We reduce the number of degrees of freedom by one as it is done in 
# EnsembleKalmanProcesses.jl
eigvals = s**2 / (ens_size-1)
Main.U = U[:,:-1]
Main.eigvals = eigvals[:-1]

Main.eval("""
    internal_cov_numpy = SVD(U, eigvals, U')
""")

true_cov = noise_ens @ noise_ens.T / (ens_size-1)
Main.true_cov = true_cov

# Main.eval("""
# println("=== Original ===")
# println(internal_cov.U)
# println(internal_cov.S)
# println(internal_cov.Vt)
# """)

# Main.eval("""
# println("=== numpy ===")
# println(internal_cov_numpy.U)
# println(internal_cov_numpy.S)
# println(internal_cov_numpy.Vt)
# """)

Main.eval("""
C1 = Matrix(internal_cov)
C2 = Matrix(internal_cov_numpy)

println("=== Matrix difference diagnostics ===")
println("frobenius norm of diff: ", norm(C1 - C2))
println("max abs entry: ", maximum(abs.(C1 - C2)))

println("frobenius norm of diff C1 - true: ", norm(C1 - true_cov))
println("frobenius norm of diff C2 - true: ", norm(C2 - true_cov))
""")

Main.eval("""
println("=== Eigenvalue comparison ===")
println("internal_cov S:        ", internal_cov.S)
println("internal_cov_numpy S: ", internal_cov_numpy.S)
println("diff:                 ",
        internal_cov.S - internal_cov_numpy.S)
""")

Main.eval("""
println("=== U column comparison (sign-adjusted) ===")

U1 = internal_cov.U
U2 = internal_cov_numpy.U

for k in 1:size(U1,2)
    # detect sign ambiguity
    sgn = sign(dot(U1[:,k], U2[:,k]))
    sgn == 0 && (sgn = 1)

    col_diff = norm(U1[:,k] - sgn * U2[:,k])

    println(
        "column ", k, ": ",
        "‖u₁ - sign·u₂‖₂ = ", col_diff,
        "   sign = ", sgn
    )
end
""")
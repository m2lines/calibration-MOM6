using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Plots
using Statistics

prior_1 = constrained_gaussian("Smagorinsky", 0.045, 0.045/2., 0.0, 0.09)
prior_2 = constrained_gaussian("ZB", 1.5, 0.75, 0.0, 3.0)
prior = combine_distributions([prior_1, prior_2])

plt = plot(prior)
display(plt)
savefig(plt, "prior/contstrained.png")

plt = plot(prior, constrained=false)
display(plt)
savefig(plt, "prior/uncontstrained.png")

N_ensemble = 100000
# Generate initial ensemble in constrained space
initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
# Transform to unconstrained space
initial_ensemble = transform_unconstrained_to_constrained(prior, initial_ensemble)
# Verify sample means and stds
println("Mean of prior constrained = ", mean(initial_ensemble; dims=2))
println("Std  of priorconstrained = ", std(initial_ensemble; dims=2))

##
using Plots
using MAT
using Distributed
using IterTools
# Import all necessary files
addprocs(85)
@everywhere include("src/problems.jl")
@everywhere include("src/algorithms.jl")

# problem1 = logistic_regression_problem(data, labels)

data = matread("ext_data/train_data_cox.mat")
const a_full = data["data_rna_train"]
const δ_full = vec(data["delta_censoring_train"])
const t_full = vec(data["daysurv_train"])
origin_n = length(δ_full)
# Generate random indices to select a third of the data
indices = sort(rand(1:origin_n, Int(origin_n / 3)))
const a = a_full[:, indices]
const δ = δ_full[indices]
const t = t_full[indices]
problem1 = cox_regression_problem(a, δ, t)
d = problem1.d
n = problem1.n
##

##
# Run experiments
# Initial value of eta
η_initial_saga = 1e-3
η_values_saga = [η_initial_saga / (1.3^i) for i in 0:8]
τ_values_saga = 0.25:0.05:0.65
parameter_pairs = collect(product(η_values_saga, τ_values_saga))
results = pmap(parameter_pairs) do pair
    η, τ = pair
    score = saga_svrg(problem1, η, τ, 1/d, 32000)
    return (η, τ, score)
end
m::Float64 = Inf
for i = 1:9
    for j = 1:9
        if results[i, j][3][2][2, end] < m
            global m = results[i, j][3][2][2, end]
        end
    end
end
for i = 1:9
    for j = 1:9
        if results[i, j][3][2][2, end] < m+0.0001
            println("parameter_pairs")
            println(results[i, j][1])
            println(results[i, j][2])
            println("final cost")
            println(m)
            gaps_saga = results[i, j][3][2]
            h = gaps_saga[1, :]
            v = gaps_saga[2, :]
            p = plot(h, v, title="Optimality Gap Over Iterations", xlabel="Iteration", ylabel="Optimality Gap", yscale=:log10, lw=2)
            display(p)
            sleep(1)
        end
    end
end
##
##
# Initial value of eta
η_initial_zpdvr = 5e-3
η_values_zpdvr = [η_initial_zpdvr / (1.25^i) for i in 0:75]
args_list_zpdvr = [(problem1, η, 1/n, 1600000) for η in η_values_zpdvr]
results = pmap((args) -> zpdvr(args...), args_list_zpdvr)
##

# Plot optimality gaps for both problems
#println("Optimality gaps for problem 1:")
#println(gaps_problem1_saga[2,:])
#=
h = gaps_problem1_saga[1, :]
v = gaps_problem1_saga[2, :]
plot(h, v, title="Optimality Gap Over Iterations", xlabel="Iteration", ylabel="Optimality Gap", yscale=:log10, lw=2)
h = gaps_problem1_zpdvr[1, :]
v = gaps_problem1_zpdvr[2, :]
plot!(h, v)
h = gaps_problem1_svrg[1, :]
v = gaps_problem1_svrg[2, :]
plot!(h, v)
=#
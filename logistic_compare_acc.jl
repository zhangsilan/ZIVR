##
using ARFFFiles, DataFrames, MLJ, GLM, CategoricalArrays
using Plots
using .Threads
using Dates
include("src/data_loader.jl")
using .DataLoader
include("src/problems.jl")
include("src/algorithms.jl")
include("src/tune_one.jl") 

# Define the data_name variable
d = 20
n = 20
L = 500.0
mu = 1e-4
#const data_name = "quadratic_n_" * string(n) * "_d_" * string(d) * "_L_" * string(L) * "_mu_" * string(mu)
const data_name = "random_n" * string(n) * "_d" * string(d)

if length(data_name)>7 && data_name[1:7] == "random_"
    X, y = load_random_logistic(d, n);
    problem1 = logistic_regression_problem(X, y);
# Load logistic regression data
elseif length(data_name)>10 && data_name[1:10] == "quadratic_"
    problem1 = quadratic_finite_sum_problem(n, d, L, mu);
else
    X, y = load_logistic_data(data_name, "./ext_data/$(data_name).arff");
    problem1 = logistic_regression_problem(X, y);
end
nothing;
##


##
# read ".ext_data/best_obj.txt" to get the best_obj, 
# if it does not exist, use proximal gradient descent to get the best_obj
find_obj = false

best_obj_file = "./ext_data/$(data_name)_l1_best_obj.txt" 
best_obj = problem1.best_obj;

if isfile(best_obj_file)
    best_obj = parse(Float64, read(best_obj_file, String))
else
    find_obj = true
end

if find_obj
    partial_svrg = (η) -> prox_svrg(problem1, η, Int64(1e5))
    η_initial = 1e-3
    i_svrg, η_svrg, best_obj_tune = tune_one_para(partial_svrg, η_initial, 1.2, 40)
    println("Index of the best stepsize: ", i_svrg)
    println("best_obj_tune", best_obj_tune)
    x, opt_gap = prox_svrg(problem1, η_svrg, Int64(1e6))
    best_obj_run = opt_gap[2, end]

    if best_obj_run == NaN
        throw(ErrorException("Stepsize not suitable"))
    end

    if best_obj_run > best_obj
        println("This run is not good enough, best_obj_run is", best_obj_run)
    else
        best_obj = best_obj_run
        open(best_obj_file, "w") do io
            write(io, string(best_obj))
        end
    end
end

problem1.best_obj = best_obj
##

##
include("src/tune_two.jl")
oracle_call=1e6
# tune stepsize for acc_zips

partial_zips = (η) -> pure_2p(problem1, η, oracle_call/2)
η_initial_zips = 5e-3/(problem1.μ*problem1.n*problem1.d)
η_zips_task = Threads.@spawn tune_one_para(partial_zips, η_initial_zips, 1.1, 40)
partial_acc_zips = (η, τ) -> acc_zips(problem1, η, τ, oracle_call/2)
i_zips, η_zips, _ = fetch(η_zips_task)
println("Index of the best stepsize for 2-point ZIPS: ", i_zips)

η_initial_acc_zips = η_zips/5
η_acc_zips_task = Threads.@spawn tune_two_para(partial_acc_zips, η_initial_acc_zips, 0, 1.1, 0.05, 50, 21)
i_acc_zips, j_acc_zips, (η_acc_zips, τ_acc_zips), result = fetch(η_acc_zips_task)
println("Best parameters for acc_zips: η = ", η_acc_zips, ", τ = ", τ_acc_zips) 
println("Best indices for acc_zips: i = ", i_acc_zips, ", j = ", j_acc_zips) 
##


##
# compare acc_zips and zips
using JLD2
oracle_call = 7e5
x_acc_zips, gaps_problem1_acc_zips = acc_zips(problem1, η_acc_zips, τ_acc_zips, oracle_call)
x_zips, gaps_problem1_zips = pure_2p(problem1, η_zips, oracle_call)
h = gaps_problem1_acc_zips[1, :]
v = gaps_problem1_acc_zips[2, :]
plot(h, v,  tickfontsize=12, title="Optimality Gap Over Iterations", xlabel="Iterations", ylabel="Optimality Gap",
xguidefontsize=13, yguidefontsize=13,yscale=:log10, lw=3, label="Acc-ZIPS", legend=:topright, legendfontsize=8)
h = gaps_problem1_zips[1, :]
v = gaps_problem1_zips[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="ZIPS")
datenow = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
savefig("./pics/$(datenow)_acc_prox_zips_vs_prox_acc_zips$(data_name).svg")
@save "./output/gaps-$(datenow)-acc_prox_zips_vs_prox_acc_zips-$(data_name).jld2" gaps_problem1_acc_zips gaps_problem1_zips
@save "./output/η-$(datenow)-acc_prox_zips_vs_prox_acc_zips-$(data_name).jld2" η_zips η_acc_zips τ_acc_zips
##
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
#const data_name = "quadratic_n_" * string(n) * "_d_" * string(d) * "_L_" * string(L) * "_mu_" * string(mu)
# const data_name = "random_n" * string(n) * "_d" * string(d)
const data_name = "cox_data"
load_params = "./output/η-a9a2025-11-13_130533.jld2"
if length(data_name)>7 && data_name[1:7] == "random_"
    X, y = load_random_logistic(d, n);
    problem1 = logistic_regression_problem(X, y);
# Load logistic regression data
elseif length(data_name)>10 && data_name[1:10] == "quadratic_"
    problem1 = quadratic_finite_sum_problem(n, d, L, mu);
elseif data_name == "cox_data"
    a, δ, t = load_cox_data("./ext_data/train_data_cox.mat")
    problem1 = cox_regression_problem(a, δ, t);
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
    partial_gd = (η) -> prox_gd(problem1, η, Int64(6e3))
    η_initial = 1e-4
    i_gd, η_gd, best_obj_tune = tune_one_para(partial_gd, η_initial, 1.2, 40)
    println("Index of the best stepsize: ", i_gd)
    println("best_obj_tune", best_obj_tune)
    x, opt_gap = prox_gd(problem1, η_gd, Int64(1e6))
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
if isfile(load_params)
    using JLD2
    η_zips, η_zpdvr, η_gradient, η_zo, η_svrg, η_acc_zips, τ_acc_zips = JLD2.load(load_params, "η_zips", "η_zpdvr", "η_gradient", "η_zo", "η_svrg", "η_acc_zips", "τ_acc_zips")
else
    oracle_call = 8e7
    partial_gradient_descent = (η) -> gradient_descent(problem1, η, oracle_call/2)
    η_initial_gradient = 1e-2
    η_gradient_task = Threads.@spawn tune_one_para(partial_gradient_descent, η_initial_gradient, 1.2, 30)

    partial_zpdvr = (η) -> zpdvr(problem1, η, 1/(problem1.n), oracle_call/2)
    η_initial_zpdvr = 1e-6
    η_zpdvr_task = Threads.@spawn tune_one_para(partial_zpdvr, η_initial_zpdvr, 1.3, 20)

    partial_zips = (η) -> pure_2p(problem1, η, oracle_call/2)
    η_initial_zips = 1e-4
    η_zips_task = Threads.@spawn tune_one_para(partial_zips, η_initial_zips, 1.1, 60)

    η_initial_zo = 1e-6
    partial_naive_zo = (η) -> naive_zo(problem1, η, oracle_call/2)
    η_zo_task = Threads.@spawn tune_one_para(partial_naive_zo, η_initial_zo, 1.3, 20)

    η_initial_svrg =  1e-5
    partial_svrg = (η) -> svrg(problem1, η, 1/(problem1.n*problem1.d), oracle_call/2)
    η_svrg_task = Threads.@spawn tune_one_para(partial_svrg, η_initial_svrg, 1.3, 20)
    # Fetch results from threads
    i_zpdvr, η_zpdvr, _ = fetch(η_zpdvr_task)
    i_gradient, η_gradient, _ = fetch(η_gradient_task)
    i_zips, η_zips, _ = fetch(η_zips_task)
    i_zo, η_zo, _ = fetch(η_zo_task)
    i_svrg, η_svrg, _ = fetch(η_svrg_task)
    println("Index of the best stepsize for ZPDVR: ", i_zpdvr)
    println("Index of the best stepsize for Gradient Descent: ", i_gradient)
    println("Index of the best stepsize for 2-point ZIPS: ", i_zips)
    println("Index of the best stepsize for Vanilla ZO: ", i_zo)
    println("Index of the best stepsize for SVRG: ", i_svrg)
end
##

##
#=
include("src/tune_two.jl")
oracle_call=2e7
# tune stepsize for acc_zips

partial_acc_zips = (η, τ) -> acc_zips(problem1, η, τ, oracle_call/2)
η_initial_acc_zips = η_zips/5
i_acc_zips, j_acc_zips, (η_acc_zips, τ_acc_zips), result = tune_two_para(partial_acc_zips, η_initial_acc_zips, 0, 1.1, 0.05, 50, 21)
println("Best parameters for acc_zips: η = ", η_acc_zips, ", τ = ", τ_acc_zips) 
println("Best indices for acc_zips: i = ", i_acc_zips, ", j = ", j_acc_zips) 
=#
##

##
using JLD2
oracle_call=2e8
zips_task = Threads.@spawn pure_2p(problem1, η_zips, oracle_call)
zpdvr_task = Threads.@spawn zpdvr(problem1, η_zpdvr, 1/problem1.n, oracle_call)
fo_task = Threads.@spawn gradient_descent(problem1, η_gradient, oracle_call)
zo_task = Threads.@spawn naive_zo(problem1, η_zo, oracle_call)
svrg_task = Threads.@spawn svrg(problem1, η_svrg, 1/(problem1.n*problem1.d), oracle_call)
acc_zips_task = Threads.@spawn acc_zips(problem1, η_acc_zips, τ_acc_zips, oracle_call)

x_dvr, gaps_problem1_dvr = fetch(zips_task)
h = gaps_problem1_dvr[1, :]
v = gaps_problem1_dvr[2, :]
plot(h, v,  tickfontsize=12, title="Optimality Gap Over Iterations", xlabel="Oracle Calls", ylabel="Optimality Gap",
xguidefontsize=13, yguidefontsize=13,yscale=:log10, lw=3, label="ZIPS(2-point)", legend=:topright, legendfontsize=8)
x_zpdvr, gaps_problem1_zpdvr = fetch(zpdvr_task)
h = gaps_problem1_zpdvr[1, :]
v = gaps_problem1_zpdvr[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="ZPDVR")
x_acc_zips, gaps_problem1_acc_zips = fetch(acc_zips_task)
h = gaps_problem1_acc_zips[1, :]
v = gaps_problem1_acc_zips[2, :]
plot!(h, v,  title="Optimality Gap Over Iterations", lw=3, label="Acc-ZIPS")
x_fo, gaps_problem1_fo = fetch(fo_task)
h = gaps_problem1_fo[1, :]
v = gaps_problem1_fo[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="Full Batch ZO")
x_zo, gaps_problem1_zo = fetch(zo_task)
h = gaps_problem1_zo[1, :]
v = gaps_problem1_zo[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="Vanilla ZO")
x_svrg, gaps_problem1_svrg = fetch(svrg_task)
h = gaps_problem1_svrg[1, :]
v = gaps_problem1_svrg[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="SVRG")
# Save plot to a file and rename by time
datenow = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
savefig("./pics/$(data_name)$(datenow).svg")
@save "./output/gaps-$(data_name)$(datenow).jld2" gaps_problem1_dvr gaps_problem1_zpdvr gaps_problem1_fo gaps_problem1_zo gaps_problem1_svrg gaps_problem1_acc_zips
@save "./output/η-$(data_name)$(datenow).jld2" η_zips η_zpdvr η_gradient η_zo η_svrg η_acc_zips τ_acc_zips
##
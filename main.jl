##
using Plots
using MAT
using Distributed
using IterTools
using .DataLoader

# Import all necessary files
addprocs(90)
@everywhere include("src/problems.jl")
@everywhere include("src/algorithms.jl")

# Load data for logistic regression
if data_name == "logistic"
    X, y = load_logistic_data("covtype", "./ext_data/covtype.arff")
    problem1 = logistic_regression_problem(X, y)

# Load data for Cox regression
elseif data_name == "cox"
    a, δ, t = load_cox_data("./ext_data/train_data_cox.mat")
    problem1 = cox_regression_problem(a, δ, t)
end

##

@everywhere function tune_one_para(alg::Function, init_para, interval = 1.1, number = 20)
    para_values = [init_para * (interval)^i for i in 1:number]
    result = pmap(para_values) do para
        score = alg(para)
        return (para, score)
    end
    m::Float64 = Inf
    for i = 1:number
        if result[i][2][2][end] < m
            m = result[i][2][2][end]
        end
    end
    for i = 1:number
        if result[i][2][2][end] < m+(1e-15)
            return result[i][1]
        end
    end
end

partial_gradient_descent = (η) -> gradient_descent(problem1, η)
η_initial = 10/(problem1.μ*problem1.n)
η_gradient_task = @spawn tune_one_para(partial_gradient_descent, η_initial, 1.2, 20)

partial_zpdvr = (η) -> zpdvr(problem1, η, 1/problem1.n)
η_initial = 0.005/(problem1.μ*problem1.n*problem1.d)
η_zpdvr_task = @spawn tune_one_para(partial_zpdvr, η_initial, 1.5, 20)

partial_p2p = (η) -> pure_2p(problem1, η)
η_initial = 0.05/(problem1.μ*problem1.n*problem1.d)
η_p2p_task = @spawn tune_one_para(partial_p2p, η_initial, 1.2, 20)

η_initial = 0.001/(problem1.μ*problem1.n*problem1.d)
partial_naive_zo = (η) -> naive_zo(problem1, η)
η_zo_task = @spawn tune_one_para(partial_naive_zo, η_initial, 1.2, 20)


η_zpdvr = fetch(η_zpdvr_task)
η_gradient = fetch(η_gradient_task)
η_p2p = fetch(η_p2p_task)
η_zo = fetch(η_zo_task)

##
p2p_task = @spawn pure_2p(problem1, η_p2p, 3.8e6)
zpdvr_task = @spawn zpdvr(problem1, η_zpdvr, 1/problem1.n, 3.8e6)
fo_task = @spawn gradient_descent(problem1, η_gradient, 3.8e6)
zo_task = @spawn naive_zo(problem1, η_zo, 3.8e6)
x_dvr, gaps_problem1_dvr = fetch(p2p_task)
h = gaps_problem1_dvr[1, :]
v = gaps_problem1_dvr[2, :]
plot(h, v,  tickfontsize=12, title="Optimality Gap Over Iterations", xlabel="Oracle Calls", ylabel="Optimality Gap",
xguidefontsize=13, yguidefontsize=13,yscale=:log10, lw=3, label="ZIPS(2-point)", legend=:bottomleft, legendfontsize=13)
x_zpdvr, gaps_problem1_zpdvr = fetch(zpdvr_task)
h = gaps_problem1_zpdvr[1, :]
v = gaps_problem1_zpdvr[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="ZPDVR")
x_fo, gaps_problem1_fo = fetch(fo_task)
h = gaps_problem1_fo[1, :]
v = gaps_problem1_fo[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="Full Batch ZO")
x_zo, gaps_problem1_zo = fetch(zo_task)
h = gaps_problem1_zo[1, :]
v = gaps_problem1_zo[2, :]
plot!(h, v, title="Optimality Gap Over Iterations", lw=3, label="Vanilla ZO")
##


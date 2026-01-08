using Random
using LinearAlgebra

# 2-point ZO gradient estimator
#Para: x: point of eva, u: random direction, β: smoothing radius
#Returns: a scalar, note that here we do not multiply it by d
function ∇uf(f::Function, x::Vector, u::Vector, β=1e-9)
    return (f(x.+β.*u) - f(x))/β
end

# d-point ZO gradient estimator
function ∇f(f::Function, x::Vector, β=1e-9)
    d = length(x)
    result = zeros(d)
    e = zeros(d)
    for i in 1:d
        e[i] = 1
        result[i] = ∇uf(f, x, e, β)
        e[i] = 0
    end
    return result
end

# Gradient Descent with optimal gap recording
function gradient_descent(problem::OptimizationProblem, η, max_oracle_calls=Int64(4e6))
    x = copy(problem.x0)
    n, d = problem.n, problem.d
    #oracle_call = 2*n*d  # A counter to store ZO oracle calls
    oracle_call = 0
    k = 0
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    while oracle_call < max_oracle_calls
        k+=1
        g = ∇f(problem.f, x)
        oracle_call += n*d
        x = problem.prox(x.-η.*g, η)
        opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    end
    return x, opt_gap
end

function acc_zips(problem::OptimizationProblem, η, τ, max_oracle_calls=Int64(4e6))
    # Initialization
    Random.seed!(70)
    n, d = problem.n, problem.d
    x = copy(problem.x0)
    y = copy(problem.x0)
    z = copy(problem.x0)
    J = zeros(d, n) # Estimated Jacobian
    running_sum = zeros(d) # vec(sum(J, dims=2))
    k = 0
    g = zeros(d)
    plt_intv = max_oracle_calls ÷ 500  # record opt gap every plt_intv
    oracle_call = 0
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    while oracle_call < max_oracle_calls
        k+=1
        if k%plt_intv == 0
            opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
        end
        x = τ.*z .+ (1-τ).*y
        e = zeros(d)
        j = rand(1:d)
        e[j] = 1
        # Random sample
        i = rand(1:n)
        zo_g = ∇uf(problem.fs[i], x, e)
        g = zo_g*d.*e .- J[j, i]*d.*e .+ running_sum
        running_sum[j] += 1/n*(zo_g - J[j, i])
        z = problem.prox(z .- η.*g, η)
        y = τ.*z .+ (1-τ).*y
        J[j, i] = zo_g
        oracle_call += 2
    end
    return x, opt_gap
end

# Double Variance Reduction: A Smoothing Trick for Composite Optimization Problems without First-Order Gradient
function zpdvr(problem::OptimizationProblem, η, p, max_oracle_calls=Int64(4e6))
    Random.seed!(70)
    n, d = problem.n, problem.d
    x = copy(problem.x0)
    w = copy(problem.x0)
    h = zeros(d)
    rind = 0
    k = 0
    g_w = zeros(d) # ̃∇f(w_k) in the original paper
    plt_interval = max_oracle_calls ÷ 500 # record opt gap every plt_intv
    oracle_call = 0  # A counter to store ZO oracle calls
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    while oracle_call < max_oracle_calls
        if rind < p
            # Sample a random direction from N(0, I)
            u = randn(d)
            g_w = h + ∇uf(problem.f, w, u)*u - u*u'*h
            oracle_call += 2*n
        end
        if k%plt_interval == 0
            opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
        end
        u = randn(d)
        i = rand(1:n)
        g = ∇uf(problem.fs[i], x, u)*u - ∇uf(problem.fs[i], w, u)*u + g_w
        x = problem.prox(x .- η.*g, η)
        oracle_call += 4
        rind = rand()
        if rind < p
            w = copy(x)
            h = h + (∇uf(problem.f, x, u)*u - u*u'*h)/(d+2)
            oracle_call += 2*n
        end
        k+=1
    end
    return x, opt_gap
end

function svrg(problem::OptimizationProblem, η, p, max_oracle_calls=Int64(4e6))
    Random.seed!(70)
    n, d = problem.n, problem.d
    x = copy(problem.x0)
    w = copy(problem.x0)
    k = 0
    g_w = ∇f(problem.f, w)
    plt_interval = max_oracle_calls ÷ 500 # record opt gap every plt_intv
    oracle_call = 0  # A counter to store ZO oracle calls
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    while oracle_call < max_oracle_calls
        if k%plt_interval == 0
            opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
        end
        # Random direction
        e = zeros(d)
        e[rand(1:d)] = 1
        # Random sample
        i = rand(1:n)
        g = ∇uf(problem.fs[i], x, e)*e*d - ∇uf(problem.fs[i], w, e)*e*d + g_w
        oracle_call += 4
        k += 1
        x = problem.prox(x - η*g, η)
        if rand() < p
            w = copy(x)
            g_w = ∇f(problem.f, w)
            oracle_call += 2*n*d
        end
    end
    return x, opt_gap
end

function pure_2p(problem::OptimizationProblem, η::Float64, max_oracle_calls=Int64(4e6))
    Random.seed!(70)
    n, d = problem.n, problem.d
    x = copy(problem.x0)
    J = zeros(d, n) # Estimated Jacobian
    running_sum = zeros(d) # vec(sum(J, dims=2))
    k = 0
    g = zeros(d)
    plt_intv = max_oracle_calls ÷ 500# record opt gap every plt_intv
    oracle_call = 0
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    e = zeros(d)
    while oracle_call < max_oracle_calls
        k+=1
        if k%plt_intv == 0
            opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
        end
        j = rand(1:d)
        e[j] = 1
        # Random sample
        i = rand(1:n)
        zo_g = ∇uf(problem.fs[i], x, e)
        g = zo_g*d.*e .- J[j, i]*d.*e .+ running_sum
        running_sum[j] += 1/n*(zo_g - J[j, i])
        x = problem.prox(x .- η.*g, η)
        J[j, i] = zo_g
        e[j] = 0
        oracle_call += 2
    end

    return x, opt_gap
end

function naive_zo(problem::OptimizationProblem, η::Float64, max_oracle_calls=Int64(4e6))
    Random.seed!(70)
    n, d = problem.n, problem.d
    x = copy(problem.x0)
    plt_intv = max_oracle_calls ÷ 500 # record opt gap every plt_intv
    k = 0
    oracle_call = 0
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    while oracle_call < max_oracle_calls
        k+=1
        if k%plt_intv == 0
            opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
        end
        e = zeros(d)
        j = rand(1:d)
        e[j] = 1
        # Random sample
        i = rand(1:n)
        g = ∇uf(problem.fs[i], x, e)*d*e
        x = problem.prox(x .- η.*g, η)
        oracle_call += 2
    end

    return x, opt_gap
end

function prox_gd(problem::OptimizationProblem, η, max_iter = Int64(1e3), compute_opt = true)
    x = copy(problem.x0)
    best_obj = problem.best_obj
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    for i=1:max_iter
        g = problem.g(x)
        x = problem.prox(x .- η.*g, η)
        if i % 5000 == 0 && compute_opt
            opt_gap = hcat(opt_gap, [i; problem.F(x)])
        elseif i % 5000 == 0
            opt_gap = hcat(opt_gap, [i; problem.F(x) - best_obj])
        end
    end
    return x, opt_gap
end

function prox_svrg(problem::OptimizationProblem, η, max_iter = Int64(1e3), compute_opt = true)
    Random.seed!(70)
    x = copy(problem.x0)
    best_obj = problem.best_obj
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    n = problem.n
    # running average of svrg
    g_tilde = problem.g(x)
    # where is that batch sampled
    w = copy(problem.x0)
    for i=1:max_iter
        j = rand(1:n)
        g = problem.gs(j, x) - problem.gs(j, w) + g_tilde
        x = problem.prox(x .- η.*g, η)
        if rand()< 1/n
            w = copy(x)
            g_tilde = problem.g(w)
        end
        if i % 5000 == 0 && compute_opt
            opt_gap = hcat(opt_gap, [i; problem.F(x)])
        elseif i % 5000 == 0
            opt_gap = hcat(opt_gap, [i; problem.F(x) - best_obj])
        end
    end
    return x, opt_gap
end

function acc_prox_gd(problem::OptimizationProblem, η, τ, max_iter = Int64(1e3))
    x = copy(problem.x0)
    y = copy(problem.x0)
    z = copy(problem.x0)
    best_obj = problem.best_obj
    opt_gap = Array{Float64}(undef, 2, 0)# First row: oracle_calls, second row: corresponding opt_gaps
    opt_gap = hcat(opt_gap, [oracle_call; max(problem.F(x) - problem.best_obj, 1e-16)])
    for i=1:max_iter
        x = τ.*z .+ (1-τ).*y
        g = problem.g(x)
        z = problem.prox(z .- η.*g, η)
        y = τ.*z .+ (1-τ).*y
        if i % 5000 == 0
            opt_gap = hcat(opt_gap, [i; problem.F(x) - best_obj])
        end
    end
    return x, opt_gap
    
end

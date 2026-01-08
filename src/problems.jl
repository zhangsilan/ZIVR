using Optim
using LinearAlgebra
using Random
using ForwardDiff


# A simplified struct definition for improved type inference
mutable struct OptimizationProblem{Tfs<:Tuple, TF, Tf, Tprox, Tg, Tgs, Tx}
    fs::Tfs
    F::TF
    f::Tf 
    prox::Tprox
    g::Tg
    gs::Tgs
    n::Int64
    d::Int64
    μ::Float64
    L::Float64
    x0::Tx
    best_obj::Float64
end

# Define a keyword constructor for improved readability
function OptimizationProblem(;
    fs::Tuple,                  # Type annotation helps documentation, but type inference does the work.
    F::Function,
    f::Function,
    prox::Function,
    g::Function,
    gs::Function,
    n::Int64,
    d::Int64,
    μ::Float64,
    L::Float64,
    x0::AbstractVector,         # Use AbstractVector for flexibility (Vector or SubArray)
    best_obj::Float64 = Inf 
)
    return OptimizationProblem(fs, F, f, prox, g, gs, n, d, μ, L, x0, best_obj)
end

# Logistic regression problem
# data: 2-d array, n*d in size, each row contains a sample point
function logistic_regression_problem(data, labels)
    # Sigmoid function (hypothesis function)
    sigmoid(z) = 1 / (1 + exp(-z))

    # Cost function for a single data point
    # x_i is the data point
    # y_i is the label 
    # θ is the parameter vector
    function logistic_cost(x_i, y_i, θ)
        h = sigmoid(dot(θ, x_i))  # Calculate h_θ(x_i)
        return -y_i * log(h) - (1 - y_i) * log(1 - h)
    end

    # Gradient of the cost function for a single data point
    function logistic_gradient(x_i, y_i, θ)
        h = sigmoid(dot(θ, x_i))
        return (h - y_i) .* x_i
    end

    d = size(data, 1)
    n = size(data, 2)
    μ = 1e-4
    λ = 1e-4 # L1 regularization term
    L = 5.0
    x0 = zeros(d)  # Initial guess (zero weights)

    prox(point, η) = sign.(point) .* max.(abs.(point) .- η*λ, 0.0)


    fs = Tuple(
    x -> logistic_cost(data[:, i], labels[i], x) + μ*(x⋅x)/2
    for i in 1:n
)

    # Full cost function
    @views F = x -> 1 / n * sum(logistic_cost(data[:, i], labels[i], x) for i in 1:n) + μ * (x ⋅ x) / 2 + λ * sum(abs.(x))
    @views f = x -> 1 / n * sum(logistic_cost(data[:, i], labels[i], x) for i in 1:n) + μ * (x ⋅ x) / 2

    # gradients of individual cost functions
    @views gs = (i, x) -> logistic_gradient(data[:, i], labels[i], x)  

    # gradient of the full cost function
    function gradient(x)
        g = zeros(d)
        for i in eachindex(labels)
            g .+= 1/n * logistic_gradient(data[:, i], labels[i], x) 
        end
        g .+= μ * x
        return g
    end
    best_obj = Inf

    return OptimizationProblem(fs = fs, 
    F = F, f = f, prox = prox, g = gradient, gs = gs,
     n = n, d = d, μ = μ, L = L, x0 = x0, best_obj = best_obj)
end

# Cox regression problem
# From ZOO-ADMM: Convergence Analysis and Applications,
# covariates: ai,
# censoring: δi
# survival_time: t
function cox_regression_problem(a, δ, t)
    d = size(a, 1)
    μ = 1e-5
    λ = 1e-6 # L1 regularization term
    L = 5.0
    x0 = 0.2*ones(d)  # Initial guess (zero weights)
    t_perm = sortperm(t)
    sorted_a = a[:, t_perm]
    sorted_δ = Bool.(δ[t_perm])
    n = size(a, 2)
    real_n = sum(sorted_δ)# Real size of this problem, excluding 0 terms

    cost_functions = Vector{Function}(undef, real_n)
    added = 0 # A counter for all added subfunctions
    for i in eachindex(sorted_δ)
        # If does not survive, don't add it
        if !sorted_δ[i]
            continue
        end
        added += 1
        @views cost_functions[added] = x -> (-sorted_a[:, i]'*x + log(sum(exp.(sorted_a[:, i:end]'*x)))+ μ*(x⋅x)/2).*real_n/n # Multiply an extra weight
    end
    fs = Tuple(cost_functions)
    prox(point, η) = sign.(point) .* max.(abs.(point) .- η*λ, 0.0)
    # prox(x, η) = sign.(x/(1+η*μ)) .* max.(abs.(x/(1+η*μ)) .- λ*η, 0.0)

    # Find minimum
    function g(x)
        gradient = zeros(d)
        for i in eachindex(sorted_δ)
            # If does not survive, don't add it
            if !sorted_δ[i]
                continue
            end
            @views exps = exp.(sorted_a[:, i:end]'*x)
            @views gradient .+= (-sorted_a[:, i] .+ sum(exps'.*sorted_a[:, i:end], dims=2)/sum(exps))/n
        end
        gradient .+= μ .* x
        return gradient
    end
    function gs(x, i)
        nothing
    end
    f = x -> 1/real_n * sum(fi(x) for fi in fs)
    F = x -> 1/real_n * sum(fi(x) for fi in fs) + λ*sum(abs.(x))
    return OptimizationProblem(fs = fs, F=F, f = f, prox = prox, g = g, gs = gs, n = real_n, d = d, 
    μ = μ, L = L, x0=x0, best_obj = Inf)
end


include("./random_psd.jl")
function quadratic_finite_sum_problem(n::Int64, d::Int64, L::Float64, μ::Float64)
    # Generate a random positive semi-definite matrix Q with condition number L/μ
    Random.seed!(70)
    λ = 1e-4 # L1 regularization term
    Q = random_matrix_with_eigenvalues(L, μ, d)

    # Generate random linear terms c for each component function
    c_components = [randn(d) for _ in 1:n]

    # Define the component cost functions f_i(x) = 0.5 * x'Qx + c_i'x
    fs = (i, x) -> 0.5 * dot(x, Q * x) + dot(c_components[i], x)

    # Define the full cost function F(x) = (1/n) * sum(f_i(x))
    F = x -> (1 / n) * sum(fi(x) for fi in cost_functions) + λ * sum(abs.(x))

    # Define the function without proximal term
    f = x -> (1 / n) * sum(fi(x) for fi in cost_functions)

    # Define the gradient of the full cost function
    g = x -> Q * x + (1 / n) * sum(c_components)

    # Define the proximal operator for L1 regularization
    prox(point, η) = sign.(point) .* max.(abs.(point) .- η*λ, 0.0)

    # Generate a random initial point x0
    x0 = randn(d)

    # Define the optimal solution (for debugging or testing purposes)
    best_obj = Inf

    return OptimizationProblem(fs = fs, 
    F = F, f = f, prox = prox, g = g,
     n = n, d = d, μ = μ, L = L, x0 = x0, best_obj = best_obj)
end
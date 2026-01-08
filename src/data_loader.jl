# This file will handle data loading for different problems

module DataLoader

using ARFFFiles, DataFrames, CategoricalArrays, MAT, Random

export load_logistic_data, load_random_logistic, load_cox_data

# Function to load logistic regression data
function load_logistic_data(data_name::String, data_path::String, sample_size::Int64 = 0)
    data = ARFFFiles.load(data_path) |> DataFrame
    # Convert target variable to binary (0 and 1)
    data[!, :target] = map(x -> x == "+1" ? 1 : 0, data[!, end])
    # Transform categorical variables to numerical values
    for col_name in names(data)
        col = data[!, col_name]
        if eltype(col) <: CategoricalValue
            data[!, col_name] = map(levelcode, col)
        end
    end
    # Separate features and target
    y = data[!, :target]
    X = Matrix(select(data, Not(:target)))'  # Convert features to a matrix
    # Select only the first `sample_size` data points
    if sample_size != 0
        X = X[:, 1:sample_size]
        y = y[1:sample_size]
    end
    return X, y
end

function load_random_logistic(d::Int64, n::Int64)
    # Generate y as random binary labels
    Random.seed!(70)
    y = rand(0:1, n)
    
    # Generate X based on the value of y
    X = zeros(d, n)
    for i in 1:n
        if y[i] == 0
            # For y = 0, generate X from a Gaussian distribution with mean -1
            X[:, i] = 0.7*randn(d) .- 1
        else
            # For y = 1, generate X from a Gaussian distribution with mean +1
            X[:, i] = randn(d) .+ 1
        end
    end
    return X, y
end

# Function to load Cox regression data
function load_cox_data(data_path::String)
    data = matread(data_path)
    a = data["data_rna_train"]
    δ = vec(data["delta_censoring_train"])
    t = vec(data["daysurv_train"])
    return a, δ, t
end

end # module DataLoader
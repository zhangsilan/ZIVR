using .Threads

function tune_two_para(alg::Function, init_para1, init_para2, interval1 = 1.1, interval2 = 0.1, number1 = 20, number2 = 8)
    para_values1 = [init_para1 * (interval1)^(i - 1) for i in 1:number1]
    if init_para2 + (number2-1) * interval2 >1
        throw(ErrorException("The second parameter grid exceeds 1,
         please choose a smaller interval2 or smaller number2"))
    end
    para_values2 = [init_para2 + interval2 * (j - 1) for j in 1:number2]
    result = Matrix{Tuple}(undef, number1, number2)

    # Flatten the 2D grid into a single loop for parallelization
    Threads.@threads for idx in 1:(number1 * number2)
        i = div(idx - 1, number2) + 1  # Compute row index
        j = mod(idx - 1, number2) + 1  # Compute column index
        para1 = para_values1[i]
        para2 = para_values2[j]
        _, score = alg(para1, para2)
        result[i, j] = ((para1, para2), score[2, end])
    end

    m::Float64 = Inf
    best_i, best_j = 0, 0
    for i in 1:number1
        for j in 1:number2
            if result[i, j][2] < m
                m = result[i, j][2]
                best_i, best_j = i, j
            end
        end
    end
    if best_i == 0 || best_j == 0
        throw(ErrorException("No minimum found, with result array: $result"))
    end
    return best_i, best_j, result[best_i, best_j][1], result
end
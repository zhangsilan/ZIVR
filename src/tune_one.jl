using .Threads

function tune_one_para(alg::Function, init_para, interval = 1.1, number = 20)
    para_values = [init_para * (interval)^i for i in 1:number]
    result = Vector{Tuple}(undef, number)
    try
        Threads.@threads for i in 1:number
            para = para_values[i]
            _, score = alg(para)
            result[i] = (para, score[2, end])
        end
        m::Float64 = Inf
        best_i = 0
        for i in 1:number
            if result[i][2] < m
                m = result[i][2]
                best_i = i
            end
        end
        if best_i == 0
            throw(ErrorException("No minimum found"))
        end
        return best_i, result[best_i][1], m
    catch e
        println("Result array at the time of error: ")
        rethrow(e)
    end
end
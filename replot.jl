using JLD2
using Plots
using Dates
datadict = Dict(
    "w8a" => "added_firsrt_gaps-w8a2025-11-29_235227.jld2",
    "a9a" => "added_firsrt_gaps-a9a2025-11-30_143358.jld2",
    "cox_data" => "added_firsrt_gaps-cox_data2025-11-12_145214.jld2",
    "compare" => "masked_gaps-2025-12-02_114245-acc_prox_zips_vs_prox_acc_zips-random_n20_d20.jld2"
)
dataname = datadict["cox_data"]
colors = theme_palette(:auto)
function replot(dataname, mask_data=nothing)
    file_loc = "./output/$(dataname)"
    gaps =  JLD2.load(file_loc)
    name_to_lengend = Dict(
        "gaps_problem1_zpdvr" => Dict("name" => "ZPDVR", "color" => colors[1], "linestyle" => :solid, "markerstyle" => :circle),
        "gaps_problem1_dvr" => Dict("name" => "ZIPVR (2-point)", "color" => colors[2], "linestyle" => :dash, "markerstyle" => :star5),
        "gaps_problem1_zips" => Dict("name" => "ZIPVR (2-point)", "color" => colors[2], "linestyle" => :dash, "markerstyle" => :star5),
        "gaps_problem1_fo" => Dict( "name" => "Full Batch ZO", "color" => colors[3], "linestyle" => :dot, "markerstyle" => :diamond),
        "gaps_problem1_zo" => Dict( "name" => "Vanilla ZO", "color" => colors[7], "linestyle" => :dashdot, "markerstyle" => :utriangle),
        "gaps_problem1_svrg" => Dict( "name" => "PZSVRG", "color" => colors[6], "linestyle" => :dashdotdot, "markerstyle" => :hexagon),
        # "gaps_problem1_acc_zips" => Dict( "name" => "Acc-ZIPVR (2-point)", "color" => colors[4], "linestyle" => :dot, "markerstyle" => :pentagon)
    )
    # --- New Code for Custom Y-Ticks ---
    # 1. Determine the y-range of your data (or a safe estimate).
    # Based on "Optimality Gap" and log scale, let's assume the data spans
    # from around 1e-3 to 1e3.
    
    # 2. Create an array of tick positions. For log scale, this usually includes
    # the intermediate values (2, 3, 4, ..., 9) * 10^k.
    
    min_log = -6 # Start at 10^(-3)
    max_log = 1.0  # End at 10^(3)
    inter_log = 1
    
    # Generate ticks for 1, 2, 5 in each decade (common for log scales)
    y_ticks = Float64[]
    for k in min_log:inter_log:max_log
        push!(y_ticks, 1 * 10^k) #
    end
    
    # Filter to keep unique, sorted, and non-zero values
    y_ticks = unique(sort(filter(x -> x > 0, y_ticks)))

    first_to_plot = true
    for (key, prop) in name_to_lengend
        if !haskey(gaps, key)
            continue
        end
        h = gaps[key][1, :]
        v = gaps[key][2, :]
        legend=prop["name"]
        color=prop["color"]
        linestyle=prop["linestyle"]
        markerstyle=prop["markerstyle"]
        if !isnothing(mask_data)
            mask = (h .>= 0) .&& (h .<= mask_data)
            h = h[mask]
            v = v[mask]
        end
        if first_to_plot
            p = plot(h, v,  tickfontsize=12, xlabel="Oracle Calls", ylabel="Optimality Gap",
            xguidefontsize=13, yguidefontsize=13,yscale=:log10, lw=3, y_ticks=y_ticks, foreground_color_legend=nothing, color=color,
            rightmargin=7Plots.Measures.mm, 
            linestyle=linestyle, label=legend, legend=:bottomleft, legendfontsize=12, background_color_legend=nothing)
            first_to_plot = false
        else
            plot!(h, v, lw=3, linestyle=linestyle, color=color, label=legend)
        end
    end
    savefig("./pics/$(dataname)-replot$(Dates.now()).svg")
end
replot(dataname, 6e7)
using Flux

include("CobraDistill.jl")
include("nnviz.jl")

cachedir = "cache/exp_data/"
n_samples = 3696

name_list = [
    "new_data_test",
]

for name in name_list
    # Plot with validation data
    #model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn = get_nn(name)
    y, ŷ = evaluate_nn_cache(nn, n_samples, cachedir)
    
    output_file =  open("cobra_" * name * ".out", "a")
    write.([output_file], string.(y) .* "\n")
    close(output_file)
    output_file =  open("eval_" * name * ".out", "a")
    write.([output_file], string.(ŷ) .* "\n")
    close(output_file)
    create_eval_plots(name, y, ŷ)

    # Calculate mean absolute error
    stats = get_stats(name)
    println(get_absolute_error(stats, 10))
end
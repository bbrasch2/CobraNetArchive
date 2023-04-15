using Flux

include("CobraDistill.jl")
include("nnviz.jl")

cachedir = "cache/rejection_random_0/"
n_samples = 10000

name_list = [
    "hypertest_lr_0.005_decay_0.999_start_1000",
    "full_optimal"
]

for name in name_list
    # Plot with validation data
    #model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    #nn = get_nn(name)
    #y, ŷ = evaluate_nn_cache(nn, n_samples, cachedir)
    #println(mean(abs.(ŷ - y)))
    
    #output_file =  open("cobra_" * name * ".out", "a")
    #write.([output_file], string.(y) .* "\n")
    #close(output_file)
    #output_file =  open("eval_" * name * ".out", "a")
    #write.([output_file], string.(ŷ) .* "\n")
    #close(output_file)

    #create_eval_plots(name, y, ŷ)

    # Calculate mean absolute error
    stats = get_stats(name)
    println(get_absolute_error(stats, 10))
end

#for lr in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
#    for decay in [0.99, 0.995, 0.999, 0.9995, 0.9999]
#        name = "lr_" * string(lr) * "_decay_" * string(decay)
#        
#        # Calculate mean absolute error
#        stats = get_stats(name)
#        println(get_absolute_error(stats, 10))
#    end
#end
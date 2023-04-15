using Flux

include("StreptoNet.jl")
include("streptoviz.jl")

hyper_list = [
    #make_hyper([64, 32, 16], 1000, 1, 0.1, 1e-3, 0.99, 0, 1e-4, 1e-5, 0.25, "dropout", "SSA_aerobic_experimental_data", "full_optimal"),
    #make_hyper([64, 32, 16], 1000, 1, 0.1, 1e-3, 0.99, 0, 1e-4, 1e-5, 0.1, "dropout_less", "SSA_aerobic_experimental_data", "full_optimal"),
    #make_hyper([64, 32, 16], 1000, 1, 0.1, 1e-3, 0.99, 0, 0, 0, 0.1, "dropout_less_noreg", "SSA_aerobic_experimental_data", "full_optimal"),
    make_hyper([16, 16, 16], 1000, 1, 0.1, 1e-3, 0.99, 0, 1e-4, 1e-5, 0, "small", "SSA_aerobic_experimental_data", "full_optimal"),
    make_hyper([16, 16, 16], 1000, 1, 0.1, 1e-3, 0.99, 0, 1e-4, 1e-5, 0.1, "small_dropout", "SSA_aerobic_experimental_data", "full_optimal"),
    make_hyper([32, 32, 32], 1000, 1, 0.1, 1e-3, 0.99, 0, 1e-4, 1e-5, 0, "medium", "SSA_aerobic_experimental_data", "full_optimal"),
    make_hyper([32, 32, 32], 1000, 1, 0.1, 1e-3, 0.99, 0, 1e-4, 1e-5, 0.1, "medium_dropout", "SSA_aerobic_experimental_data", "full_optimal"),
]

for hyper in hyper_list
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper)
    train_nn(hyper, nn, stats, epoch_path, epoch_skips)
    create_run_plots(hyper.rundir)
end

#for decay in [0.99]
#    hyper = make_hyper([64, 32, 16], 1000, 1, 0.1, 1e-3, decay, 250, 2e-4, 2e-5, "decay_" * string(decay) * "_fixedreg2", "SSA_aerobic_experimental_data", "full_optimal")
#    nn, stats, epoch_path, epoch_skips = get_training_status(hyper)
#    train_nn(hyper, nn, stats, epoch_path, epoch_skips)
#    create_run_plots(hyper.rundir)
#    #train_error, test_error = get_absolute_error(get_stats(hyper.rundir), 10)
#    #println("Train error: " * string(train_error))
#    #println("Test error: " * string(test_error))
#end
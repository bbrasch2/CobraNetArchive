using Flux

include("StreptoNet.jl")
include("streptoviz.jl")

hyper_list = [
    make_hyper([64, 32, 16, 8], 1000, 8, 0.1, Flux.ADAM, 1e-3, 0.995, 100, 1e-5, 1e-5, 0.1, "relu_test", "SSA_aerobic_experimental_data", "hypertest_lr_0.005_decay_0.999_start_1000"),
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
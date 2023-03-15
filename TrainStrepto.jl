using Flux

include("StreptoNet.jl")
include("streptoviz.jl")

hyper_list = [
    make_hyper([256, 128, 64], 500, 1, 0.1, 1e-2, 1, 0, "lr2", "SSA_aerobic_experimental_data", "decay_test2"),
    make_hyper([256, 128, 64], 500, 1, 0.1, 1e-3, 1, 0, "lr3", "SSA_aerobic_experimental_data", "decay_test2"),
    make_hyper([256, 128, 64], 500, 1, 0.1, 1e-4, 1, 0, "lr4", "SSA_aerobic_experimental_data", "decay_test2"),
    make_hyper([256, 128, 64], 500, 1, 0.1, 1e-5, 1, 0, "lr5", "SSA_aerobic_experimental_data", "decay_test2"),
    make_hyper([256, 128, 64], 500, 1, 0.1, 1e-6, 1, 0, "lr6", "SSA_aerobic_experimental_data", "decay_test2"),
]

for hyper in hyper_list
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper)
    train_nn(hyper, nn, stats, epoch_path, epoch_skips)
    create_run_plots(hyper.rundir)
end
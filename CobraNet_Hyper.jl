# nohup julia --project=.venv/CobraNet CobraNet_Hyper.jl &> cobranet_hyper_sigmoid.out &
# nohup julia --project=.venv/CobraNet CobraNet_Hyper.jl &> cobranet_hyper_relu.out &

using Flux

include("StreptoNet.jl")
include("streptoviz.jl")

# Original
#learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
#decays = [0.98, 0.99, 0.995, 0.999]
#starts = [0, 100, 200]
#l1s = [1e-6, 1e-5, 1e-4, 1e-3]
#l2s = [1e-6, 1e-5, 1e-4]
#dropouts = [0, 0.1, 0.2]
#names = []

# Regularization
#learning_rates = [1e-3]
#decays = [0.995]
#starts = [100]
#l1s = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
#l2s = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
#dropouts = [0, 0.1, 0.2, 0.3]
#names = []

# Learning rates
learning_rates = [1e-4, 1e-3, 1e-2]
decays = [0.99, 0.995, 0.999, 0.9995]
starts = [0, 100, 200]
l1s = [1e-9]
#l1s = [1e-8]
l2s = [1e-7]
#l2s = [1e-5]
dropouts = [0.1]
names = []

for lr in learning_rates
    for decay in decays
        for start in starts
            for l1 in l1s
                for l2 in l2s
                    for dropout in dropouts
                        #name = "hypertest_lr_" * string(lr) * "_decay_" * string(decay) * "_start_" * string(start) * "_l1_" * string(l1) * "_l2_" * string(l2) * "_dropout_" * string(dropout)
                        name = "lrtest_sigmoid_lr_" * string(lr) * "_decay_" * string(decay) * "_start_" * string(start)
                        #name = "lrtest_relu_lr_" * string(lr) * "_decay_" * string(decay) * "_start_" * string(start)
                        push!(names, name)
                        hyper = make_hyper([64, 32, 16, 8],sigmoid,1000,8,0.1,Flux.ADAM,lr,decay,start,l1,l2,dropout,name,"SSA_aerobic_experimental_data","hypertest2_lr_0.005_decay_0.999_start_0")
                        #hyper = make_hyper([64, 32, 16, 8],relu,1000,8,0.1,Flux.ADAM,lr,decay,start,l1,l2,dropout,name,"SSA_aerobic_experimental_data","hypertest2_lr_0.005_decay_0.999_start_0")
                        nn, stats, epoch_path, epoch_skips = get_training_status(hyper)
                        train_nn(hyper, nn, stats, epoch_path, epoch_skips)
                        create_run_plots(hyper.rundir)
                    end
                end
            end
        end
    end
end

for name in names
    train_error, test_error = get_absolute_error(get_stats(name), 10)
    println(name * ": " * string(test_error))
end

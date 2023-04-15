# nohup julia --project=.venv/CobraNet CobraNet_Arch.jl &> cobranet_arch.out &

using Flux

include("StreptoNet.jl")
include("streptoviz.jl")

arches = [
    [4],
    [4, 3],
    [4, 3, 2],
    [4, 3, 2, 1],
    [4, 3, 2, 1, 0]
]
sizes = [0, 1, 2, 3, 4]
optimizers = [Flux.Descent, Flux.ADAM, Flux.NADAM]
names = []

for arch in arches
    for size in sizes
        for optimizer in optimizers
            name = "archtest_layer_" * string(length(arch)) * "_size_" * string(size) * "_opt_" * string(optimizer)
            push!(names, name)
            hyper = make_hyper(2 .^ (arch .+ size),1000,8,0.1,optimizer,5e-3,1,0,1e-4,1e-5,0,name,"SSA_aerobic_experimental_data","hypertest_lr_0.005_decay_0.999_start_1000")
            nn, stats, epoch_path, epoch_skips = get_training_status(hyper)
            train_nn(hyper, nn, stats, epoch_path, epoch_skips)
            create_run_plots(hyper.rundir)
        end
    end
end

for name in names
    train_error, test_error = get_absolute_error(get_stats(name), 10)
    println(name * ": " * string(test_error))
end
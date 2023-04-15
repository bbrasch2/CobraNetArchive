# nohup julia --project=.venv/CobraNet Surrogate_Arch.jl &> surrogate_arch.out &

using Flux

include("CobraDistill.jl")
include("nnviz.jl")

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
            hyper = make_hyper(2 .^ (arch .+ size),fill(elu, length(arch) + 1),6000,128,optimizer,5e-3,1,0,name,"random")
            model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
            nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
            train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
            create_run_plots(hyper.rundir)
        end
    end
end

for name in names
    stats = get_stats(name)
    println(name * ": " * string(get_absolute_error(stats, 10)))
end
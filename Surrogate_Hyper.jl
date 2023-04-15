# nohup julia --project=.venv/CobraNet Surrogate_Hyper.jl &> surrogate_hyper.out &

using Flux

include("CobraDistill.jl")
include("nnviz.jl")

learning_rates = [1e-3, 5e-3, 1e-2]
decays = [0.99, 0.995, 0.999, 0.9995, 0.9999, 1]
starts = [0, 1000, 2000]
names = []

for lr in learning_rates
    for decay in decays
        for start in starts
            name = "hypertest_lr_" * string(lr) * "_decay_" * string(decay) * "_start_" * string(start)
            push!(names, name)
            hyper = make_hyper([32, 16, 8, 4],fill(elu,5),6000,128,Flux.NADAM,lr,decay,start,name,"random")
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
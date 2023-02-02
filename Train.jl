using Flux

include("CobraDistill.jl")
include("nnviz.jl")

hyper_list = [
    make_hyper([256, 128, 64],[elu, elu, elu, elu],4000,128,0.995,"nadam_decay5","rejection_random_split"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],4000,128,0.99,"nadam_decay6","rejection_random_split"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],4000,128,0.95,"nadam_decay7","rejection_random_split"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],4000,128,0.9,"nadam_decay8","rejection_random_split"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],4000,128,0.5,"nadam_decay9","rejection_random_split"),
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir, plot_lr=false)
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    y, ŷ = evaluate_nn(get_nn(hyper.rundir), oracle, model, binvars, convars, 100)
    create_eval_plots(hyper.rundir, y, ŷ)
end
using Flux

include("CobraDistill.jl")
include("nnviz.jl")

hyper_list = [
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,1e-4,"new_data_test2","rejection_random_80"),
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
using Flux

include("CobraDistill.jl")
include("nnviz.jl")

hyper_list = [
    make_hyper([512, 256, 128, 64],[elu, elu, elu, elu, elu],6000,128,2e-3,0.999,0,"split_50_big","rejection_random_50")
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir)
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    y, ŷ = evaluate_nn(get_nn(hyper.rundir), oracle, model, binvars, convars, 100)
    create_eval_plots(hyper.rundir, y, ŷ)
end
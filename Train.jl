include("CobraDistill.jl")
include("nnviz.jl")

# TODO: Separate resume-training into new file so params can be changed

hyper_list = [
    make_hyper([256, 128, 64],[elu, elu, elu, elu],1000,1,1e-6,"test_lr1"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],1000,1,1e-7,"test_lr2"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],1000,1,1e-8,"test_lr3")
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper)
end
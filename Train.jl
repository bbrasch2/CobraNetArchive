using Flux

include("CobraDistill.jl")
include("nnviz.jl")

# TODO: Separate resume-training into new file so params can be changed

hyper_list = [
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],500,1,ExpDecay(1e-5,1,1,1e-10,1),"test_space7"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],500,1,ExpDecay(1e-6,1,1,1e-10,1),"test_space8"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],500,1,ExpDecay(1e-7,1,1,1e-10,1),"test_space9"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],500,1,ExpDecay(1e-8,1,1,1e-10,1),"test_space10"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],500,1,ExpDecay(1e-9,1,1,1e-10,1),"test_space11")
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir)
end
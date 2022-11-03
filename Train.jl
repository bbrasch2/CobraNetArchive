using Flux

include("CobraDistill.jl")
include("nnviz.jl")

# TODO: Separate resume-training into new file so params can be changed

hyper_list = [
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],2500,1,ExpDecay(1e-1,1,1,1e-10,1),"testdecay18"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],2500,1,ExpDecay(1e-2,1,1,1e-10,1),"testdecay19"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],2500,1,ExpDecay(1e-3,1,1,1e-10,1),"testdecay20"),
    make_hyper([1024, 512, 256],[elu, elu, elu, elu],2500,1,ExpDecay(1e-4,1,1,1e-10,1),"testdecay21"),
    make_hyper([2048, 512, 256],[elu, elu, elu, elu],2000,1,ExpDecay(1e-2,0.9925,1,1e-10,1),"testdecay22"),
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir)
end
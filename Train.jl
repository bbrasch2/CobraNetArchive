using Flux

include("CobraDistill.jl")
include("nnviz.jl")

hyper_list = [
    make_hyper([128, 64, 32],[elu, elu, elu, elu],15000,128,5e-3,0.9995,0,"full_optimal","random")
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir)
    #model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    #y, ŷ = evaluate_nn(get_nn(hyper.rundir), oracle, model, binvars, convars, 100)
    #create_eval_plots(hyper.rundir, y, ŷ)
end

#for mix in 0:10:100
#    hyper = make_hyper([128, 64, 32],[elu, elu, elu, elu],9000,128,5e-3,0.9995,0,"mix_" * string(mix),"rejection_random_" * string(mix))
#    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
#    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
#    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
#    create_run_plots(hyper.rundir)
#end
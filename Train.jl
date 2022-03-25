include("CobraDistill.jl")

hyper = make_hyper([256, 128, 64],[elu, elu, elu, elu],10,1,1e-7,"test1")
model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
stats, run_path, epoch_path = make_stats(hyper)
nn = make_nn(hyper,ntotal)
train_nn(hyper,nn,oracle,model,binvars,convars)





# Hyper is now made inside a function
# Test data is now generated inside a function
# Stats are initiated by a function
# NN is made inside function
# NN training is done inside function
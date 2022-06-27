include("CobraDistill.jl")
include("nnviz.jl")

hyper_list = [
    make_hyper([256, 128, 64], [elu, elu, elu, elu],2000,1,ExpDecay(1e-2,0.995,1,1e-10,1),"testdecay9")
]

for hyper in hyper_list
    create_run_plots(hyper)
end
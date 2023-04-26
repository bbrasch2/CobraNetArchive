include("nnviz.jl")

name_list = [
    "hypertest_lr_0.005_decay_0.999_start_1000"
]

for name in name_list
    create_run_plots(name)
end
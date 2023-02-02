include("nnviz.jl")

name_list = [
    "nadam_decay4"
]

for name in name_list
    create_run_plots(name, plot_lr=false)
end
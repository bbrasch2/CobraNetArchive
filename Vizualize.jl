include("nnviz.jl")

name_list = [
    "NAdam_opt_optimal2"
]

for name in name_list
    create_run_plots(name, plot_lr=false)
end
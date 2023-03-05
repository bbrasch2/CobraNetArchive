include("nnviz.jl")

name_list = [
    "NAdam_CCD_opt"
]

for name in name_list
    create_run_plots(name, plot_lr=false)
end
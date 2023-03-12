include("nnviz.jl")

name_list = [
    "NAdam_CCD_1_1"
]

for name in name_list
    create_run_plots(name, plot_lr=false)
end
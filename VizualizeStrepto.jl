include("streptoviz.jl")

name_list = [
    "hypertest_lr_0.01_decay_0.98_start_0_l1_0.001_l2_0.0001_dropout_0.0"
]

for name in name_list
    create_run_plots(name)
end
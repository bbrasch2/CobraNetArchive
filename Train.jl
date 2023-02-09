using Flux

include("CobraDistill.jl")
include("nnviz.jl")

hyper_list = [
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.99), 0.0000001),"NAdam_opt_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.99), 0.00000001),"NAdam_opt_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.99), 0.000000001),"NAdam_opt_3","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.999), 0.0000001),"NAdam_opt_4","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.999), 0.00000001),"NAdam_opt_5","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.999), 0.000000001),"NAdam_opt_6","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.9999), 0.0000001),"NAdam_opt_7","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.9999), 0.00000001),"NAdam_opt_8","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.8, 0.9999), 0.000000001),"NAdam_opt_9","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.99), 0.0000001),"NAdam_opt_10","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.99), 0.00000001),"NAdam_opt_11","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.99), 0.000000001),"NAdam_opt_12","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.999), 0.0000001),"NAdam_opt_13","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.999), 0.00000001),"NAdam_opt_14","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.999), 0.000000001),"NAdam_opt_15","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.9999), 0.0000001),"NAdam_opt_16","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.9999), 0.00000001),"NAdam_opt_17","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.9, 0.9999), 0.000000001),"NAdam_opt_18","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.99), 0.0000001),"NAdam_opt_19","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.99), 0.00000001),"NAdam_opt_20","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.99), 0.000000001),"NAdam_opt_21","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.999), 0.0000001),"NAdam_opt_22","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.999), 0.00000001),"NAdam_opt_23","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.999), 0.000000001),"NAdam_opt_24","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.9999), 0.0000001),"NAdam_opt_25","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.9999), 0.00000001),"NAdam_opt_26","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.001, (0.99, 0.9999), 0.000000001),"NAdam_opt_27","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.99), 0.0000001),"NAdam_opt_28","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.99), 0.00000001),"NAdam_opt_29","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.99), 0.000000001),"NAdam_opt_30","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.999), 0.0000001),"NAdam_opt_31","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.999), 0.00000001),"NAdam_opt_32","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.999), 0.000000001),"NAdam_opt_33","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.9999), 0.0000001),"NAdam_opt_34","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.9999), 0.00000001),"NAdam_opt_35","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.8, 0.9999), 0.000000001),"NAdam_opt_36","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.99), 0.0000001),"NAdam_opt_37","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.99), 0.00000001),"NAdam_opt_38","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.99), 0.000000001),"NAdam_opt_39","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.999), 0.0000001),"NAdam_opt_40","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.999), 0.00000001),"NAdam_opt_41","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.999), 0.000000001),"NAdam_opt_42","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.9999), 0.0000001),"NAdam_opt_43","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.9999), 0.00000001),"NAdam_opt_44","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.9, 0.9999), 0.000000001),"NAdam_opt_45","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.99), 0.0000001),"NAdam_opt_46","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.99), 0.00000001),"NAdam_opt_47","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.99), 0.000000001),"NAdam_opt_48","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.999), 0.0000001),"NAdam_opt_49","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.999), 0.00000001),"NAdam_opt_50","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.999), 0.000000001),"NAdam_opt_51","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.9999), 0.0000001),"NAdam_opt_52","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.9999), 0.00000001),"NAdam_opt_53","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],2000,128,(0.0001, (0.99, 0.9999), 0.000000001),"NAdam_opt_54","rejection_random_80"),
]

for hyper in hyper_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir, plot_lr=false)
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    y, ŷ = evaluate_nn(get_nn(hyper.rundir), oracle, model, binvars, convars, 100)
    create_eval_plots(hyper.rundir, y, ŷ)
end
using Distributed
addprocs(4; exeflags=`--project=$(Base.active_project())`)  # add 4 worker processes

#@everywhere using Pkg
#@everywhere Pkg.instantiate()
@everywhere using Flux
@everywhere include("CobraDistill.jl")
@everywhere include("nnviz.jl")

hyper_list = [
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(36, (0.00501187233627272, 0.920567176527572), 0.99),"NAdam_CCD_1_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(15, (0.0001, 0.999), 0.9999),"NAdam_CCD_2_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(22, (0.0001, 0.920567176527572), 0.9999),"NAdam_CCD_3_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(25, (0.00501187233627272, 0.920567176527572), 0.9999),"NAdam_CCD_4_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(27, (0.00501187233627272, 0.999), 0.99),"NAdam_CCD_5_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(31, (0.00501187233627272, 0.999), 0.99),"NAdam_CCD_6_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(24, (0.00501187233627272, 0.920567176527572), 0.9999),"NAdam_CCD_7_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(4, (0.0001, 0.920567176527572), 0.99),"NAdam_CCD_8_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(35, (0.0001, 0.999), 0.9999),"NAdam_CCD_9_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(20, (0.0001, 0.999), 0.9999),"NAdam_CCD_10_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(11, (0.0001, 0.999), 0.99),"NAdam_CCD_11_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(16, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_12_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(17, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_13_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(46, (0.00501187233627272, 0.999), 0.9999),"NAdam_CCD_14_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(37, (0.00501187233627272, 0.920567176527572), 0.9999),"NAdam_CCD_15_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(52, (0.00501187233627272, 0.920567176527572), 0.9999),"NAdam_CCD_16_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(38, (0.00501187233627272, 0.999), 0.9999),"NAdam_CCD_17_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(39, (0.0001, 0.920567176527572), 0.9999),"NAdam_CCD_18_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(33, (0.00501187233627272, 0.920567176527572), 0.99),"NAdam_CCD_19_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(3, (0.00501187233627272, 0.999), 0.99),"NAdam_CCD_20_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(34, (0.0001, 0.920567176527572), 0.9999),"NAdam_CCD_21_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(53, (0.0001, 0.999), 0.99),"NAdam_CCD_22_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(21, (0.00501187233627272, 0.999), 0.9999),"NAdam_CCD_23_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(50, (0.0001, 0.999), 0.99),"NAdam_CCD_24_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(49, (0.0001, 0.920567176527572), 0.99),"NAdam_CCD_25_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(23, (0.00501187233627272, 0.920567176527572), 0.9999),"NAdam_CCD_26_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(10, (0.00501187233627272, 0.920567176527572), 0.99),"NAdam_CCD_27_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(28, (0.0001, 0.999), 0.99),"NAdam_CCD_28_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(14, (0.00501187233627272, 0.920567176527572), 0.99),"NAdam_CCD_29_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(45, (0.00501187233627272, 0.999), 0.99),"NAdam_CCD_30_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(32, (0.00501187233627272, 0.999), 0.9999),"NAdam_CCD_31_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(5, (0.00501187233627272, 0.920567176527572), 0.99),"NAdam_CCD_32_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(12, (0.0001, 0.999), 0.99),"NAdam_CCD_33_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(7, (0.0001, 0.920567176527572), 0.9999),"NAdam_CCD_34_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(26, (0.0001, 0.920567176527572), 0.99),"NAdam_CCD_35_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(9, (0.0001, 0.920567176527572), 0.9999),"NAdam_CCD_36_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(42, (0.00501187233627272, 0.999), 0.99),"NAdam_CCD_37_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(41, (0.00501187233627272, 0.920567176527572), 0.99),"NAdam_CCD_38_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(43, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_39_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(2, (0.0001, 0.999), 0.99),"NAdam_CCD_40_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(30, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_41_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(18, (0.00501187233627272, 0.999), 0.99),"NAdam_CCD_42_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(1, (0.0001, 0.920567176527572), 0.9999),"NAdam_CCD_43_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(51, (0.0001, 0.999), 0.9999),"NAdam_CCD_44_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(48, (0.0001, 0.999), 0.9999),"NAdam_CCD_45_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(44, (0.00501187233627272, 0.920567176527572), 0.9999),"NAdam_CCD_46_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(19, (0.0001, 0.920567176527572), 0.99),"NAdam_CCD_47_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(29, (0.00501187233627272, 0.999), 0.9999),"NAdam_CCD_48_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(6, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_49_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(13, (0.0001, 0.999), 0.9999),"NAdam_CCD_50_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(40, (0.0001, 0.920567176527572), 0.99),"NAdam_CCD_51_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(8, (0.0001, 0.920567176527572), 0.99),"NAdam_CCD_52_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(47, (0.00501187233627272, 0.999), 0.9999),"NAdam_CCD_53_1","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(22, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_1_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(7, (0.000707945784384138, 0.99988779815457), 0.999),"NAdam_CCD_2_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(8, (0.000707945784384138, 0.991087490618663), 0.99999),"NAdam_CCD_3_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(15, (0.0354813389233575, 0.991087490618663), 0.999),"NAdam_CCD_4_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(19, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_5_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(28, (0.000707945784384138, 0.292054215615862), 0.999),"NAdam_CCD_6_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(11, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_7_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(4, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_8_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(1, (0.000707945784384138, 0.991087490618663), 0.9),"NAdam_CCD_9_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(18, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_10_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(23, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_11_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(14, (0.000707945784384138, 0.292054215615862), 0.999),"NAdam_CCD_12_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(17, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_13_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(24, (0.000707945784384138, 0.991087490618663), 0.99999),"NAdam_CCD_14_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(10, (0.000707945784384138, 0.991087490618663), 0.99999),"NAdam_CCD_15_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(26, (1.41253754462275E-05, 0.991087490618663), 0.999),"NAdam_CCD_16_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(6, (0.000707945784384138, 0.292054215615862), 0.999),"NAdam_CCD_17_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(12, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_18_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(2, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_19_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(3, (1.41253754462275E-05, 0.991087490618663), 0.999),"NAdam_CCD_20_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(20, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_21_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(9, (0.0354813389233575, 0.991087490618663), 0.999),"NAdam_CCD_22_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(16, (0.000707945784384138, 0.991087490618663), 0.9),"NAdam_CCD_23_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(25, (0.000707945784384138, 0.991087490618663), 0.999),"NAdam_CCD_24_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(29, (0.000707945784384138, 0.991087490618663), 0.9),"NAdam_CCD_25_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(5, (1.41253754462275E-05, 0.991087490618663), 0.999),"NAdam_CCD_26_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(13, (0.000707945784384138, 0.99988779815457), 0.999),"NAdam_CCD_27_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(21, (0.000707945784384138, 0.99988779815457), 0.999),"NAdam_CCD_28_2","rejection_random_80"),
    make_hyper([256, 128, 64],[elu, elu, elu, elu],3000,128,(27, (0.0354813389233575, 0.991087490618663), 0.999),"NAdam_CCD_29_2","rejection_random_80"),
]

# Define a function to be executed in parallel
@everywhere function run_hyper(hyper, output_file)
    io = open(output_file, "w")
    redirect_stdout(io)
    redirect_stderr(io)
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn, stats, epoch_path, epoch_skips = get_training_status(hyper,ntotal)
    train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    create_run_plots(hyper.rundir, plot_lr=false)
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    y, ŷ = evaluate_nn(get_nn(hyper.rundir), oracle, model, binvars, convars, 100)
    create_eval_plots(hyper.rundir, y, ŷ)
    close(io)
end

# Call the function in parallel for each hyperparameter setting
@sync @distributed for i = 1:length(hyper_list)
    # Create a file for each worker's output
    output_file = "training_$(i).out"
    # Call the run_hyper function for the i-th hyperparameter setting and redirect the output to the i-th file
    run_hyper(hyper_list[i], output_file)
end

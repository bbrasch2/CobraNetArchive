using Flux, BSON, DataFrames

name_list = [
    "NAdam_opt_5",
    "NAdam_opt_6",
    "NAdam_opt_7",
    "NAdam_opt_8",
    "NAdam_opt_16",
    "NAdam_opt_17",
    "NAdam_opt_optimal",
    "NAdam_opt_optimal2",
]

for name in name_list
    run_dir = "runs/" * name * "/"
    epoch_dir = run_dir * "epochs/"
    eval_dir = run_dir * "eval/"
    img_dir = run_dir * "img/"
    hyper_file = run_dir * "hyper.txt"
    archive_dir = "runs/archive/" * name * "/"

    # Get most recent epoch file
    max_epoch = 0
    most_recent_saved = ""
    for bsonfile in readdir(epoch_dir)
        filename = epoch_dir * bsonfile
        epoch = parse(Int, splitext(splitdir(filename)[2])[1])
        if epoch > max_epoch
            max_epoch = epoch
            most_recent_saved = filename
        end
    end
    bson_file = most_recent_saved
    
    # Copy BSON file, eval_dir, img_dir, and hyper.txt to archive directory
    mkpath(archive_dir)
    cp(bson_file, archive_dir * "data.bson")
    if isdir(eval_dir)
        cp(eval_dir, archive_dir * "eval/")
    end
    if isdir(img_dir)
        cp(img_dir, archive_dir * "img/")
    end
    cp(hyper_file, archive_dir * "hyper.txt")

    # Delete original run directory
    rm(run_dir, recursive=true)
end
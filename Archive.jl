using Flux, BSON, DataFrames

name_list = [
    "nadam_decay",
    "nadam_decay2",
    "nadam_decay3",
    "nadam_decay4",
    "nadam_decay5",
    "nadam_decay6",
    "nadam_decay7",
    "nadam_decay8",
    "nadam_decay9",
    "nadam_lr1",
    "nadam_lr2",
    "nadam_lr3",
    "rejection_nadam",
    "rejection_nadam2",
    "rejection_nadam3",
    "rejection_nadam4",
    "rejection_nadam10",
    "rejection_nadam11",
    "rejection_nadam12",
    "rejection_nadam13",
    "rejection_random_70",
    "rejection_random_70_big",
    "rejection_random_80"
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
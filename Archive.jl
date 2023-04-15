using Flux, BSON, DataFrames

name_list = [
    "random_size_0",
    "random_size_1",
    "random_size_2",
    "random_size_3",
    "random_size_4",
    "random_size_5",
    "random_size_6",
    "random_size_7",
    "spacefill_size_0",
    "spacefill_size_1",
    "spacefill_size_2",
    "spacefill_size_3",
    "spacefill_size_4",
    "spacefill_size_5",
    "spacefill_size_6",
    "spacefill_size_7",
    "rejection_random_50_size_0",
    "rejection_random_50_size_1",
    "rejection_random_50_size_2",
    "rejection_random_50_size_3",
    "rejection_random_50_size_4",
    "rejection_random_50_size_5",
    "rejection_random_50_size_6",
    "rejection_random_50_size_7",
]

for lr in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    for decay in [0.99, 0.995, 0.999, 0.9995, 0.9999]
        name = "lr_" * string(lr) * "_decay_" * string(decay)

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
end
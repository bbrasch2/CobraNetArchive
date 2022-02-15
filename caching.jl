using JLD

function cache_training_data(n_per_file, n_files, sampler, dir)
    if !endswith(dir, "/")
        dir *= "/"
    end
    mkpath(dir)
    for file = 1:n_files
        X, y = sampler(n_per_file)
        filename = dir * string(file) * ".jld"
        save(filename, "X", X, "y", y)
        println("Finished file ", string(file), " of ", string(n_files))
    end
end

function get_batch(cachedir, n; skip=0)
    files = readdir(cachedir, join=true)
    println("loading file " * files[1])
    Xin, yin = load(files[1], "X", "y")
    m = size(Xin, 1)
    Xout = similar(Xin, m, n)
    yout = similar(yin, n)
    to_skip = skip
    needed = n
    current_out = 1
    for i = 1:length(files)
        if i > 1
            # for n=1 we've already loaded the file above
            println("loading file " * files[i])
            Xin, yin = load(files[i], "X", "y")
        end
        available = size(Xin, 2)
        current_in = 1
        if to_skip > 0
            # we're still skipping over samples
            if to_skip ≥ available
                # we need to skip this entire file
                to_skip -= available
                continue
            else
                # we need to skip part of this file
                current_in += to_skip
                available -= to_skip
                to_skip = 0
            end
        end

        if available ≥ needed
            idx_out = current_out:(current_out+needed-1)
            idx_in = current_in:(current_in+needed-1)
            Xout[:,idx_out] = Xin[:,idx_in]
            yout[idx_out] = yin[idx_in]
            return Xout, yout
        else # available < needed
            # take the rest of the file
            idx_out = current_out:(current_out+available-1)
            Xout[:,idx_out] = Xin[:,current_in:end]
            yout[idx_out] = yin[current_in:end]
            needed -= available
        end
    end
end

function serve_batches(ch::Channel, cachedir, n, nbatches; skip=0)
    files = readdir(cachedir, join=true)

    # load the first file to get the sizes of the arrays
    println("loading file " * files[1])
    Xin, yin = load(files[1], "X", "y")
    current_in = 1
    available = size(Xin, 2)

    # initialize the output arrays
    m = size(Xin, 1) # number of features
    Xout = similar(Xin, m, n)
    yout = similar(yin, n)
    current_out = 1

    to_skip = skip
    needed = n
    served = 0
    for i = 1:length(files)
        if i > 1
            # for n=1 we've already loaded the file above
            println("loading file " * files[i])
            Xin, yin = load(files[i], "X", "y")
            current_in = 1
            available = size(Xin, 2)
        end

        if to_skip > 0
            # we're still skipping over samples
            if to_skip ≥ available
                # we need to skip this entire file
                to_skip -= available
                continue
            else
                # we need to skip part of this file
                current_in += to_skip
                available -= to_skip
                to_skip = 0
            end
        end

        while available > 0 && served < nbatches
            if available ≥ needed
                # serve another batch
                idx_out = current_out:(current_out+needed-1)
                idx_in = current_in:(current_in+needed-1)
                Xout[:,idx_out] = Xin[:,idx_in]
                yout[idx_out] = yin[idx_in]
                put!(ch, (Xout, yout))
                served += 1
                available -= needed
                current_in += needed
                current_out = 1
                needed = n
            else # available < needed
                # take the rest of the file
                idx_out = current_out:(current_out+available-1)
                Xout[:,idx_out] = Xin[:,current_in:end]
                yout[idx_out] = yin[current_in:end]
                needed -= available
                current_out += available
            end
        end
        if served ≥ nbatches
            break
        end
    end
    if served < nbatches
        error("not enough batches in cache")
    end
end


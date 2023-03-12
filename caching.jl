using JLD
using CSV
using DataFrames

function cache_training_data(n_per_file, n_files, sampler, dir)
    if !endswith(dir, "/")
        dir *= "/"
    end
    mkpath(dir)
    for file = 1:n_files
        filename = dir * string(file) * ".jld"
        if isfile(filename)
            println("Found file ", string(file), " of ", string(n_files))
            flush(stdout)
            continue
        end
        X, y = sampler(n_per_file)
        save(filename, "X", X, "y", y)
        println("Finished file ", string(file), " of ", string(n_files))
        flush(stdout)
    end
end

function mix_cache(cache_input1, cache_input2, cache_output, mix_ratio, n_files)
    # Validate inputs
    if !endswith(cache_output, "/")
        cache_output *= "/"
    end
    mkpath(cache_output)
    if !endswith(cache_input1, "/")
        cache_input1 *= "/"
    end
    if !endswith(cache_input2, "/")
        cache_input2 *= "/"
    end

    for epoch = 1:n_files
        # Skip if output file exists already
        output_filename = cache_output * string(epoch) * ".jld"
        if isfile(output_filename)
            println("Found file ", string(epoch), " of ", string(n_files))
            flush(stdout)
            continue
        end
        
        # Skip any missing inputs
        input_filename1 = cache_input1 * string(epoch) * ".jld"
        input_filename2 = cache_input2 * string(epoch) * ".jld"
        if !isfile(input_filename1) || !isfile(input_filename2)
            println("One of the two input files is missing for file ", string(epoch))
            flush(stdout)
            continue
        end

        # Load data
        X1, y1 = load(input_filename1, "X", "y")
        X2, y2 = load(input_filename2, "X", "y")

        # Reject any dimension mismatch
        if !(size(X1)[2] == size(X2)[2] == size(y1)[1] == size(y2)[1])
            println("Dimension mismatch for file ", string(epoch))
            flush(stdout)
            continue
        end

        # Mix data
        n_replace = Int(floor(size(X1)[2] * (1 - mix_ratio)))
        locs = Random.randperm(size(X1)[2])[1:n_replace]
        X1[:,locs], y1[locs] = X2[:,locs], y2[locs]
        save(output_filename, "X", X1, "y", y1)
        
        println("Finished file ", string(epoch), " of ", string(n_files))
        flush(stdout)
    end
end

# Produce a cache file based on a CSV import of a specific format
function AAs_from_csv(csv_filepath, exch_rxns_filepath, sampler, cache_path)
    if isfile(cache_path)
        println("Found file ", cache_path)
        return
    end

    data = DataFrame(CSV.File(csv_filepath))
    binvars = chop.(readlines(exch_rxns_filepath), tail=5)
    AA_subset = data[!, binvars]

    # TODO: this might be problem line - this might lose the positional data of
    # the AAs, which means the oracle just sets the first 20 binvals
    binvals = Matrix{Float32}(AA_subset)'

    X, y = sampler(binvals)

    save(cache_path, "X", X, "y", y)
    println("Finished file ", cache_path)
    return
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


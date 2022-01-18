

n_epoch = 5

function create_training_data(ch::Channel)
    for i = 1:n_epoch
        epoch = string(i)
        println("Starting create_training_data " * epoch)
        sleep(5)
        println("Finished training data " * epoch)
        put!(ch, "epoch " * epoch)
    end
end

function save_stats(ch::Channel)
    while true
        epoch = take!(ch)
        println("Starting save_stats on " * epoch)
        sleep(10)
        println("Finished save_stats on " * epoch)
    end
end

ch_data = Channel(1)
ch_save = Channel(1)

Threads.@spawn create_training_data(ch_data)
Threads.@spawn save_stats(ch_save)

for epoch in ch_data
    println("Starting training on epoch " * epoch)
    sleep(3)
    println("Finished training on epoch " * epoch)
    put!(ch_save, epoch)
end

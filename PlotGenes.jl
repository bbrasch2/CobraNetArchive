using BSON, CSV, DataFrames, LinearAlgebra, Flux, Plots, Plots.PlotMeasures

ENV["GKSwstype"] = "100"

name = "lrtest_sigmoid_lr_0.01_decay_0.995_start_0"
epoch = "1000"
exchange_names = "iSMU_amino_acid_exchanges.txt"
gene_names = "iSMU_amino_acid_genes.txt"
output = "gene_predictions/" * name

bson = BSON.load("streptoruns/" * name * "/epochs/" * epoch * ".bson")

AAs = bson["X"]'
genes = bson["genes"]'
data = hcat(AAs, genes)
columns = [readlines(exchange_names)..., readlines(gene_names)...]

CSV.write(output * ".csv", DataFrame(data, columns))

hm = heatmap(1:size(genes,2), 1:size(genes,1), genes, # c=cgrad([:white, :green]), 
    xticks=(1:size(genes,2),readlines(gene_names)),
    xrotation = 90, size = (2000, 3000), margin = 60px,
    tickfontsize=24, xtickfontsize=8)#, titlefontsize=36, title=name)
savefig(output * ".png")

gene_norm = zeros(size(genes))
for i in 1:size(gene_norm, 2)
    d = maximum(genes[:,i]) - minimum(genes[:,i])
    if d == 0
        d += 1
    end
    gene_norm[:,i] = (genes[:,i] .- minimum(genes[:,i])) ./ d
end
data_norm = hcat(AAs, gene_norm)

CSV.write(output * "_norm.csv", DataFrame(data_norm, columns))

hm = heatmap(1:size(gene_norm,2), 1:size(gene_norm,1), gene_norm, # c=cgrad([:white, :green]), 
    xticks=(1:size(gene_norm,2),readlines(gene_names)),
    xrotation = 90, size = (2000, 3000), margin = 60px,
    tickfontsize=24, xtickfontsize=8)#, titlefontsize=36, title=name * "_norm")
savefig(output * "_norm.png")
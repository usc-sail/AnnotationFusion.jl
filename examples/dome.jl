using Pkg
Pkg.activate(@__DIR__) # Activating environment
Pkg.instantiate() # Downloads packages if needed

using CSV
using Glob
using Random
using Statistics
using DataFrames
using Distributions
using LinearAlgebra
using AnnotationFusion

path = joinpath(dirname(@__DIR__), "data/DOME/")
file = joinpath(path, "dome_annotations_M1.csv")
df = CSV.read(file, DataFrame)

function aggregate_row(row; method::AnnotationFusion.FusionMethod = Copeland(), annotators = ["A$i" for i in 1:3], persons = 1:4)

    columns = vcat(["Person", annotators]...)
    rankings = hcat([Vector(row[[Symbol(string(a, i)) for i in 1:4]]) for a in annotators]...)
    annotations = DataFrame(hcat(persons, rankings), columns)
    aggregates = fuse(annotations, :Person, method)

    return DataFrame([Symbol("P$(i)_$(name(method))") for i in persons] .=> aggregates)

end

merged_copeland = vcat(map(row -> aggregate_row(row; method=Copeland()), eachrow(df))...)
merged_mean = vcat(map(row -> aggregate_row(row; method=Mean()), eachrow(df))...)

df = hcat(df, merged_mean, merged_copeland)

CSV.write(joinpath(path, "dome_annotations_M1_merged.csv"), df)
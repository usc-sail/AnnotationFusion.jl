using CSV
using Plots
using DataFrames
using AnnotationFusion

file = joinpath(dirname(dirname(@__FILE__)), "data/behavioral_codes.csv")
data = CSV.read(file, DataFrame)

raters = :annotatorID
items = :sessionID # Contains info on columns {stage, initiator, person_rated}
ratings = :dim1

annotations = dense(data, raters, items, ratings)

μ = fuse(annotations, :sessionID, Mean())

te = fuse(annotations, :sessionID, TE())

plot(μ); plot!(te)
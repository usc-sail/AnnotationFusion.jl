using Pkg
Pkg.activate(@__DIR__) # Activating environment
Pkg.instantiate() # Downloads packages if needed

using CSV
using DataFrames
using AnnotationFusion

path = joinpath(dirname(@__DIR__), "data/COACH/")
savepath = joinpath(path, "aggregates")
mkpath(savepath)

file = joinpath(path, "COACH_scores.csv")
df = CSV.read(file, DataFrame)

raters = :ANNOTID
items = :SESSIONID
scores = [:SCO01, :SCO02, :SCO03, :SCO04, :SCO05]

## Initial preprocessing
# Consider only cases where a transcript is available
filter!(row -> row.Transcript == true, df)
anonymize!(df, raters)

for score in scores
    aggregates = Dict{String, Vector{Float64}}()
    annotations = dense(df, raters, items, score)

    for method in [Mean(), Median(), Copeland(scaling=:distribution), Copeland(scaling=:procrustes), TE(ntriplets=:all, scaling=:procrustes), TE(ntriplets=:all, scaling=:distribution)]
        aggregates[name(method)] = fuse(annotations, items, method)
        # if occursin("te", name(method))
        #     # aggregates[name(method)] = clamp.(aggregates[name(method)], 1, 7)
        # end
    end

   CSV.write(joinpath(savepath, string("COACH_", score, ".csv")), hcat(annotations[:,[items]], DataFrame(aggregates)))
end
function scale(reference::Array{Float64,1}, X::Array{Float64,1})
    return scale(reshape(reference, size(reference,1),1), reshape(X, size(X,1),1))
end

function scale(reference::Array{Float64,2}, X::Array{Float64,2})
    # We solve the scaling problem by min || aX - reference - b||^2,
    # where (a,b) are the scale and offset parameters

    @assert size(reference,2) == 1
    @assert size(reference) == size(X)

    n::Int64 = size(X,1)
    a::Float64 = dot(X,reference) / (dot(X,X) - sum(X)/(n * sum(reference)))
    b::Float64 = (a * sum(X) - sum(reference)) / n

    return a*X - b
end

function CCC(X::Array{Float64,2}, Y::Array{Float64,2})::Float64
    if size(X,2) == 1 & size(Y,2) == 1
        X = reshape(X, size(X,1), )
        Y = reshape(Y, size(Y,1), )
        return CCC(X,Y)
    else
        warn("Check dimensions")
    end
end

function CCC(X::Array{Float64,1}, Y::Array{Float64,1})::Float64
    ρ = 2cov(X, Y, corrected=true) / (var(X) + var(Y) + (mean(X) - mean(Y)) ^ 2)
end

function CCCweights(annotations::DataFrame)
    # Precompute the number of annotators
    nannotators = size(annotations, 2)

    CCCs = zeros(Float64, nannotators, nannotators)

    for i in 1:nannotators, j in 1:nannotators
        CCCs[i,j] = CCC(annotations[:,i], annotations[:,j])
    end

    return sum(CCCs, dims=2) .- 1
end

function agreements(triplets1::T, triplets2::T) where T <: Triplets

    n1, n2 = maximum(triplets1), maximum(triplets2)

    n1 == n2 || throw(ArgumentError("Number of items in each triplet set does not match"))

    intersection = intersect(Set(triplets1), Set(triplets2))

    return length(intersection) / ( n1 * binomial(n1 - 1, 2))
end

function dense(data::DataFrame, raters::Symbol, items::Symbol, ratings::Symbol)
    groups = groupby(select(data, [items, raters, ratings]), raters)
    df = select(convert(DataFrame, groups[1]), [items, ratings])

    for i = 2:length(groups)
        aux = select(convert(DataFrame, groups[i]), [items, ratings])
        df = outerjoin(df, aux, on=items, makeunique=true)
    end

    annotators = unique(data[!, raters])
    rename!(df, [items; Symbol.(annotators)])

    return df
end


function columncounts(df::AbstractDataFrame, column::Symbol)
    return Dict([(i, count(x -> x == i, df[!,column])) for i in unique(df[!,column])])
end

"""
    function anonymize(df::DataFrame, raters::Symbol)

Anonymize the raters' names in columns `raters`. Returns
the same dataframe `df` with hashed values in the
`raters` column.
"""
function anonymize(df::DataFrame, raters::Symbol)
    annotators = unique(df[!,raters])
    hashed = Dict(annotators .=> hashsha256.(annotators))
    df[!,raters] = [hashed[annotator] for annotator in df[!,raters]]
    return df
end

function anonymize!(df::DataFrame, raters::Symbol)
    annotators = unique(df[!,raters])
    hashed = Dict(annotators .=> hashsha256.(annotators))
    df[!,raters] = [hashed[annotator] for annotator in df[!,raters]]
end

function hashsha256(s::AbstractString; nchars::Int = 8)
    return bytes2hex(sha256(s))[1:nchars]
end

function anonymize(df::DataFrame)
    annotators = names(df)
    hashed = Dict(annotators .=> hashsha256.(annotators))
    rename!(df, hashed)
    return df
end

"""
    function name(method::FusionMethod)

Return the name of an annotation fusion method from its signature.

This method should be used to create column names in a DataFrame.
"""
function name(method::FusionMethod)
    if :scaling in fieldnames(typeof(method))
        return lowercase(string(typeof(method), "_", method.scaling))
    else
        return lowercase(string(typeof(method)))
    end
end

"""
    function fillmissing!(row, value::T) where T <: Real

Fill missing values for a vector, using `value` to replace the missing values..
"""
function fillmissing!(row, value::T) where T <: Real
    for i in eachindex(row)
        row[i] = ismissing(row[i]) ? value : row[i]
    end
end

"""
    function fillmissing!(df::DataFrame, f::Function)

Fill missing values per row, using the function f applied row-wise.
"""
function fillmissing(df::DataFrame, index::Symbol, f::Function)
    imputed = copy(df)

    for name in names(imputed[!, Not(index)])
        imputed[!, name] = convert(Vector{Union{Float64,Missing}}, imputed[!, name])
    end

    for row in eachrow(imputed)
        fillmissing!(row[Not(index)], f(row[Not(index)]))
    end

    return imputed
end

function fillmissing(df::AbstractMatrix, f::Function)
    imputed = convert(Matrix{Union{Missing, Float64}}, df)

    for row in eachrow(imputed)
        fillmissing!(row, f(row))
    end

    return imputed
end
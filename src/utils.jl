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

function agreements(
    triplets1::TripletEmbeddings.Triplets{Tuple{Int64,Int64,Int64}},
    triplets2::TripletEmbeddings.Triplets{Tuple{Int64,Int64,Int64}})

    n1 = maximum(getindex.(triplets1, [1 2 3]))
    n2 = maximum(getindex.(triplets1, [1 2 3]))

    n1 == n2 || throw(ArgumentError("Number of items does not match"))

    set1 = Set(triplets1)
    set2 = Set(triplets2)

    return length(intersect(set1, set2)) / ( n1 * binomial(n1 - 1, 2))
end
function Triplets(
    annotations::DataFrame,
    index::Symbol,
    ntriplets::Int,
    weights::Vector{Float64} = ones(Float64, size(annotations[:, Not(index)], 2))
    )

    nannotations, nannotators = size(annotations[:, Not(index)])
    samples = sampletriplets(nannotations, ntriplets)
    counter = 0

    S = TripletEmbeddings.tripletstype(nannotations)
    triplets = Vector{Triplet{S}}(undef, 0)
    D = [distances(column) for column in eachcol(annotations[:, Not(index)])]

    for t in samples
        less_than = 0
        greater_than = 0

        for l in 1:nannotators
            if !ismissing(D[l][t[1], t[2]]) && !ismissing(D[l][t[1], t[3]])
                if D[l][t[1], t[2]] < D[l][t[1], t[3]]
                    less_than += weights[l]
                elseif D[l][t[1], t[2]] > D[l][t[1], t[3]]
                    greater_than += weights[l]
                end
            end
        end

        if less_than > greater_than
            counter += 1
            push!(triplets, Triplet(t[1], t[2], t[3]))
        elseif less_than < greater_than
            counter += 1
            push!(triplets, Triplet(t[1], t[3], t[2]))
        end
    end

    return triplets
end

function distances(annotations::Vector{Union{T, Missing}}) where T <: Real
    n = length(annotations)
    D = zeros(Union{Float64,Missing}, n, n)

    for i in 1:n, j in i+1:n
        if !ismissing(annotations[i]) && !ismissing(annotations[j])
            D[i,j] = sqeuclidean(annotations[i], annotations[j])
        else
            D[i,j] = missing
        end
    end

    return D + D'
end

function distances(annotations::Vector{T}) where T <: Real
    return pairwise(SqEuclidean(), annotations', dims=2)
end

function Triplets(Y::AbstractMatrix{T}, n::Int) where T <: Real
    # triplets = [Triplets(Y[:, partition]) for partition in Iterators.partition(1:size(Y, 2), n)]
    for partition in Iterators.partition(1:size(Y, 2), n)
        println(partition)
        Triplets(Y[:, partition])
        # println(partition)
    end
end
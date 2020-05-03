function Triplets(
    annotations::DataFrame;
    weights::Array{Float64,1}=ones(Float64,size(annotations,2))
    )

    nannotations = size(annotations, 1)
    ntriplets = nannotations * binomial(nannotations - 1, 2)
    nannotators = size(annotations, 2) - 1
    triplets = Vector{Tuple{Int,Int,Int}}(undef, ntriplets)
    counter = 0

    D = [distances(annotations[:,i]) for i in 1:nannotators]

    for k = 1:nannotations, j = 1:k-1, i = 1:nannotations
        if i != j && i != k

            less_than = 0
            greater_than = 0

            for l = 1:nannotators
                if !ismissing(D[l][i,j]) && !ismissing(D[l][i,k])
                    if D[l][i,j] < D[l][i,k]
                        less_than += weights[l]
                    elseif D[l][i,j] > D[l][i,k]
                        greater_than += weights[l]
                    end
                end
            end

            if less_than > greater_than
                counter += 1
                @inbounds triplets[counter] = (i, j, k)
            elseif less_than < greater_than
                counter += 1
                @inbounds triplets[counter] = (i, k, j)
            end
        end
    end

    return Triplets(triplets[1:counter])
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
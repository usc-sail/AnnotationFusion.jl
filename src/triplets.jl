function Triplets(
    annotations::DataFrame;
    weights::Vector{Float64}=ones(Float64, size(annotations, 2))
)

    C = 40
    nannotations, nannotators = size(annotations)
    ntriplets = round(Int, C*nannotations*log(nannotations))
    triplets = Vector{Tuple{Int,Int,Int}}(undef, ntriplets)

    counter = 0

    # D = [distances(annotations[:,i]) for i in 1:nannotators]
    is = rand(1:nannotations, ntriplets)

    for t = 1:ntriplets
        # println(t)
        i = is[t]
        k = rand(setdiff(2:nannotations, i))
        j = rand(setdiff(1:k-1, i))

        less_than = 0
        greater_than = 0

        for l = 1:nannotators
            # First check that none of the annotations are missing
            if all([!ismissing(annotations[idx,l]) for idx in [i,j,k]])
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
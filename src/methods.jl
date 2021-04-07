abstract type FusionMethod end

"""
    Mean <: FusionMethod

Use the mean to combine annotations.

# Constructor

    function Mean(; f = x -> x)

Construct a Mean method. f is the function applied when taking the mean. For example:

```julia
julia> using Statistics

julia> mean(√, [1, 2, 3])
1.3820881233139908
```
"""
struct Mean <: FusionMethod
    f::Function

    function Mean(; f = x -> x)
        new(f)
    end
end

"""
    fuse(annotations::DataFrame, index::Symbol, method::Mean[; g::Function=x -> x])

Fuse annotations by applying the mean over annotations. This assumes an annotations matrix where each row contains
several annotations for an item or session.

# Parameters

 - g is a function to be applied _after_ the mean is computed. Can be used for rounding or casting.

# Examples
```julia
julia> annotations = DataFrame(items = string.(1:3), annotator1 = 1:3, annotator2 = 2:4, annotator3 = 3:5)
3×4 DataFrame
 Row │ items   annotator1  annotator2  annotator3
     │ String  Int64       Int64       Int64
─────┼────────────────────────────────────────────
   1 │ 1                1           2           3
   2 │ 2                2           3           4
   3 │ 3                3           4           5

julia> fuse(annotations, :items, Mean)
3×5 DataFrame
 Row │ items   annotator1  annotator2  annotator3  mean
     │ String  Int64       Int64       Int64       Float64
─────┼─────────────────────────────────────────────────────
   1 │ 1                1           2           3      2.0
   2 │ 2                2           3           4      3.0
   3 │ 3                3           4           5      4.0

julia> fuse(annotations, :items, Mean(); g = x -> round(Int, x))
3×5 DataFrame
 Row │ items   annotator1  annotator2  annotator3  mean
     │ String  Int64       Int64       Int64       Int64
─────┼───────────────────────────────────────────────────
   1 │ 1                1           2           3      2
   2 │ 2                2           3           4      3
   3 │ 3                3           4           5      4
```

"""
function fuse!(annotations::DataFrame, index::Symbol, method::Mean; g::Function=x -> x)
    annotations.mean = fuse(annotations, index, method; g=g)
    return annotations
end

function fuse(annotations::DataFrame, index::Symbol, method::Mean; g::Function=x -> x)
    return [g(mean(method.f, skipmissing(row[Not(index)]))) for row in eachrow(annotations)]
end

function fuse(annotations::AbstractMatrix, method::Mean; g::Function=x -> x)
    return [g(mean(method.f, skipmissing(row))) for row in eachrow(annotations)]
end

"""
    Mean <: FusionMethod

Use the mean to combine annotations.

# Constructor

    function Mean(; f = x -> x)

Construct a Mean method. f is the function applied when taking the mean. For example:

```julia
julia> using Statistics

julia> mean(√, [1, 2, 3])
1.3820881233139908
```
"""
struct Median <: FusionMethod end

"""
    fuse(annotations::DataFrame, index::Symbol, method::Median[; g::Function=x -> x])

Fuse annotations by applying the median over annotations. This assumes an annotations matrix where each row contains
several annotations for an item or session.

# Parameters

 - g is a function to be applied _after_ the mean is computed. Can be used for rounding or casting.

# Examples
```julia
julia> annotations = DataFrame(items = string.(1:3), annotator1 = 1:3, annotator2 = 2:4, annotator3 = 3:5)
3×4 DataFrame
 Row │ items   annotator1  annotator2  annotator3
     │ String  Int64       Int64       Int64
─────┼────────────────────────────────────────────
   1 │ 1                1           2           3
   2 │ 2                2           3           4
   3 │ 3                3           4           5

julia> fuse(annotations, :items, Median())
3×5 DataFrame
 Row │ items   annotator1  annotator2  annotator3  mean
     │ String  Int64       Int64       Int64       Float64
─────┼─────────────────────────────────────────────────────
   1 │ 1                1           2           3      2.0
   2 │ 2                2           3           4      3.0
   3 │ 3                3           4           5      4.0

julia> fuse(annotations, :items, Median(); g = x -> round(Int, x))
3×5 DataFrame
 Row │ items   annotator1  annotator2  annotator3  mean
     │ String  Int64       Int64       Int64       Int64
─────┼───────────────────────────────────────────────────
   1 │ 1                1           2           3      2
   2 │ 2                2           3           4      3
   3 │ 3                3           4           5      4
```

"""
function fuse!(annotations::DataFrame, index::Symbol, method::Median; g::Function=x -> x)
    annotations.mean = [g(median(skipmissing(row[Not(index)]))) for row in eachrow(annotations)]
    return annotations
end

function fuse(annotations::DataFrame, index::Symbol, method::Median; g::Function=x -> x)
    return [g(median(skipmissing(row[Not(index)]))) for row in eachrow(annotations)]
end

function fuse(annotations::AbstractMatrix, method::Median; g::Function=x -> x)
    return [g(median(skipmissing(row))) for row in eachrow(annotations)]
end

"""
    TE([loss::T=tSTE(α=30), ntriplets::Symbol = :all, verbose::Bool=true, print_every::Int=50]) where T <: TripletEmbeddings.AbstractLoss

Create a Triplet Embedding fusion method.

# Arguments

 - loss: Loss to be used (between STE and tSTE). Default is `tSTE(α=30)` (equivalent to `STE()` but more numerically stable)
 - ntriplets: Number of triplets to sample between `[:all, :auto]`. Use :auto if you run into memory issues, it randomly samples 20nlog(n) triplets.
 - scaling: Scaling of the embeddings. Options: `[:distribution, :procrustes]`. Use `:procrustes` for continuous scales, `:distribution` for discrete scales.

# Summary

struct TE <: AnnotationFusion.FusionMethod

# Fields

    loss        :: TripletEmbeddings.AbstractLoss
    ntriplets   :: Symbol
    verbose     :: Bool
    print_every :: Int64

# Supertype Hierarchy

TE <: AnnotationFusion.FusionMethod <: Any
"""
struct TE <: FusionMethod
    loss::T where T <: TripletEmbeddings.AbstractLoss
    ntriplets::Symbol
    constant::Int
    scaling::Symbol
    verbose::Bool
    print_every::Int

    function TE(; loss::T=tSTE(α=30), ntriplets::Symbol = :all, constant::Int = 40, scaling::Symbol=:procrustes, verbose::Bool=true, print_every::Int=50) where T <: TripletEmbeddings.AbstractLoss
        ntriplets in [:all, :auto] || throw(ArgumentError("ntriplets must be one of [:auto, :all]"))
        constant ≥ 1 || throw(ArgumentError("Constant in Cnlog(n) must be ≥ 1."))
        scaling in [:distribution, :procrustes] || throw(ArgumentError("scaling must be one of [:distribution, :procrustes]"))
        new(loss, ntriplets, constant, scaling, verbose, print_every)
    end
end

function fuse(annotations::DataFrame, index::Symbol, method::TE)
    n = size(annotations, 1)

    ntriplets = if method.ntriplets == :auto
        # If auto, uses the min between all triplets and method.constant * nlog(n)
        floor(Int, min(size(annotations, 1) * binomial(size(annotations, 1) - 1, 2), method.constant * size(annotations, 1) * log(size(annotations, 1))))
    elseif method.ntriplets == :all
        n * binomial(n - 1, 2)
    end

    triplets = Triplets(annotations, index, ntriplets)

    μ = fuse(annotations, index, Mean())
    X = Embedding(μ) # set the initial condition as the mean, since they are somewhat close

    misclassifications = TripletEmbeddings.fit!(method.loss, triplets, X; verbose=method.verbose, print_every=method.print_every)

    if method.scaling == :procrustes
        X, tr = procrustes(X, Matrix(μ'))
        return TripletEmbeddings.Vector(X)
    elseif method.scaling == :distribution
        return rankings_to_ratings(annotations, index, μ, TripletEmbeddings.Vector(X))
    end
end

function fuse!(annotations::DataFrame, index::Symbol, method::TE)
    annotations.TE = fuse(annotations, index, method)
end

struct Copeland <: FusionMethod
    imputation::Function
    scaling::Symbol

    """
        function Copeland([imputation::Function = row -> mean(skipmissing(row)), scaling::Symbol=:distribution])

    Create a Copeland's method struct.

    # Arguments

     - imputation: Function to impute values (per row). Defaults to the mean of the row values.
     - scaling: Either by reusing the ratings' distribution of the annotations matrix (`:distribution`) or `:procrustes`. Defaults to `:distribution`.
    """
    function Copeland(; imputation::Function = row -> mean(skipmissing(row)), scaling::Symbol=:distribution)
        scaling in [:distribution, :procrustes] || throw(ArgumentError("scaling mus be one of [:distribution, :procrustes]"))

        new(imputation, scaling)
    end
end

function copeland(annotations::DataFrame)
    points = zeros(size(annotations,1))
    nannotations, nannotators = size(annotations)

    # For each annotator, we compare all of their annotations with each other
    for i in 1:nannotations, j = i + 1:nannotations
        votesᵢ = 0 # Votes for each option
        votesⱼ = 0

        for a in 1:nannotators
            ratingᵢ = annotations[i, a]
            ratingⱼ = annotations[j, a]

             if !ismissing(ratingᵢ) || !ismissing(ratingⱼ)
                 if !ismissing(ratingᵢ) && ismissing(ratingⱼ)
                     ratingⱼ = 0
                 elseif ismissing(ratingᵢ) && ismissing(ratingⱼ)
                     ratingᵢ = 0
                 end

                if !ismissing(ratingᵢ) && !ismissing(ratingⱼ)
                    if ratingᵢ > ratingⱼ
                        votesᵢ = votesᵢ + 1
                    elseif ratingᵢ < ratingⱼ
                        votesⱼ = votesⱼ + 1
                    end
                end
             end
        end

        if votesᵢ > votesⱼ
            points[i] = points[i] + 1
        elseif votesᵢ < votesⱼ
            points[j] = points[j] + 1
        else
            points[i] = points[i] + 1/2
            points[j] = points[j] + 1/2
        end
    end

    return points
end

function fuse(annotations::DataFrame, index::Symbol, method::Copeland)
    μ = fuse(annotations, index, Mean())
    imputed = fillmissing(annotations, index, method.imputation)
    points = copeland(imputed[:, Not(index)])

    if method.scaling == :procrustes
        points, tr = procrustes(Matrix(points'), Matrix(μ'))
        return Vector(dropdims(points', dims=2))
    elseif method.scaling == :distribution
        return rankings_to_ratings(annotations, index, μ, points)
    end
end

function fuse!(annotations::DataFrame, index::Symbol, method::Copeland)
    annotations.copeland = fuse(annotations, index, method)
end

function fuse(annotations::AbstractMatrix, method::Copeland)
    μ = fuse(annotations, Mean())
    imputed = fillmissing(annotations, method.imputation)

    if method.scaling == :procrustes
        points, tr = procrustes(Matrix(points'), Matrix(μ'))
        return Vector(dropdims(points', dims=2))
    elseif method.scaling == :distribution
        return rankings_to_ratings(annotations, index, μ, points)
    end
end

function rankings_to_ratings(annotations::DataFrame, index::Symbol, means::Vector{T}, points::Vector{T}) where T <: Real
    size(annotations, 1) == length(points) || throw(ArgumentError("Number of rows in annotations must be the same as length of points"))

    results = sort(DataFrame([annotations[!,index],  points], [index, :points]), :points)

    vals = collect(skipmissing(vec(Matrix(annotations[:, Not(index)]))))
    scale = sort(unique(vals))

    counts = fit(Histogram, vals, nbins=length(scale)).weights
    ratings = zeros(Float64, length(points))
    args = ceil.(Int, cumsum(counts) ./ sum(counts) .* length(ratings))

    ratings[1:args[1]] .= scale[1]
    for i in 2:length(scale)
        ratings[args[i-1] + 1:args[i]] .= scale[i]
    end

    results.ratings = ratings
    return sort(results, index).ratings
end
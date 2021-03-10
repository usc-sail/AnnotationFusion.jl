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


struct TE <: FusionMethod
    loss::T where T <: TripletEmbeddings.AbstractLoss
    print_every::Int

    function TE(; loss::T=tSTE(α=30), print_every::Int=50) where T <: TripletEmbeddings.AbstractLoss
        # tSTE(α=30) is equivalent to STE(σ=1/√(2)), but more numerically stable
        new(loss, print_every)
    end
end

function fuse(annotations::DataFrame, index::Symbol, method::TE)
    n = size(annotations, 1)
    triplets = Triplets(annotations, index)

    μ = fuse(annotations, index, Mean())
    X = Embedding(μ) # set the initial condition as the mean
    misclassifications = fit!(method.loss, triplets, X; print_every=method.print_every)

    X, tr = procrustes(X, Matrix(μ'))

    return TripletEmbeddings.Vector(X)
end

function fuse!(annotations::DataFrame, index::Symbol, method::TE)
    annotations.TE = fuse(annotations, index, method)
end


# struct Copeland <: FusionMethod end
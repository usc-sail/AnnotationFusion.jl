# AnnotationFusion
This package implements algorithms based on Copeland's method and Triplet Embeddings for fusion of human annotations. It supports both annotations in time (i.e. regression) as well as session-level annotations (i.e. classification, or unique annotations for a given item).

# Installation
To install the latest version, use Julia 1.2 or greater. In a Julia REPL, do:
```julia
julia> ]
(@v1.4) pkg> add www.github.com/usc-sail/AnnotationFusion.jl
```

# Usage
## Helper scripts
Scripts to automatically use this package from the command line may be found in [this repo](https://www.github.com/kmundnic/annotation-fusion). These scripts will install the necessary packages and run the fusion from the annotations passed to it from CSV file.

## Package
This package may also be used directly in Julia scripts.

### Classification (session-level annotations)
We assume that the annotations are saved in a CSV file, where each row represents a session and each column contains annotations from different annotators:
```julia
│ Row  │ carlos  │ lisa    │ mae     │ soyoon  │
│      │ Int64⍰  │ Int64⍰  │ Int64⍰  │ Int64⍰  │
├──────┼─────────┼─────────┼─────────┼─────────┼
│ 1    │ 6       │ 9       │ 6       │ missing │
│ 2    │ 7       │ 9       │ 7       │ 7       │
│ 3    │ 7       │ 9       │ 7       │ 7       │
│ 4    │ 4       │ 9       │ 7       │ missing │
│ 5    │ 3       │ 3       │ 3       │ missing │
│ 6    │ 5       │ 5       │ 3       │ missing │
│ 7    │ 6       │ 5       │ 3       │ 6       │
│ 8    │ 5       │ 4       │ 2       │ 4       │
│ 9    │ 4       │ 5       │ 3       │ missing │
⋮
```
#### Triplet Embeddings
We compute the pairwise distances between columns (considering the missing values) and mine triplets over these. The number of triplets mined depends on the number of items or sessions. By default, we mine all possible triplets, but this may fail if the number of items to embed is too large (> 500 items).

Here's an example on how to run the code, using the [TripletEmbeddings.jl](www.github.com/usc-sail/TripletEmbeddings.jl) package:

```julia
using CSV
using DataFrames
using AnnotationFusion
using TripletEmbeddings

data = CSV.read("data.csv") # Or select the path where your data is

raters = :raters
items = :items
ratings = Symbol("some.answer")

# This is required is the ratings are saved in a sparse array instead of a matrix
arousal = AnnotationFusion.generateAnnotationsMatrix(data, raters, items, ratings)

# Generate the triplets from a matrix, where rows indicate items and columns indicate raters
triplets = AnnotationFusion.Triplets(arousal[!,2:end])

dims = 1
X = Embedding(dims, maximum(triplets)) # Initialize a random embedding
misclassifications = fit!(STE(), triplets, X; max_iterations=1000, verbose=false)
```
The embedding `X` will have the embedded items in `dims`-dimensional space.

#### Copeland's method

### Regression (or real-time, or time-continuous annotations)

# References
 - Mundnich K, Nasir M, Georgiou PG, Narayanan SS. _Exploiting Intra-Annotator Rating Consistency Through Copeland's Method for Estimation of Ground Truth Labels in Couples' Therapy_. In INTERSPEECH 2017 (pp. 3167-3171). [PDF](https://sail.usc.edu/publications/files/mundnichinterspeech2017.pdf)
 - Booth BM, Mundnich K, Narayanan S. _Fusing annotations with majority vote triplet embeddings_. In Proceedings of the 2018 on Audio/Visual Emotion Challenge and Workshop 2018 Oct 15 (pp. 83-89). [PDF](https://sail.usc.edu/publications/files/p83-booth.pdf)
 - Mundnich K, Booth BM, Girault B, Narayanan S. _Generating labels for regression of subjective constructs using triplet embeddings_. Pattern Recognition Letters. 2019 Dec 1; 128:385-92. [PDF](https://sail.usc.edu/publications/files/1-s2.0-S0167865519302752-main%20(1).pdf)

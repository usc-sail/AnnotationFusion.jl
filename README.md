# AnnotationFusion
This package implements algorithms based on Copeland's method and Triplet Embeddings for fusion of human annotations. It supports both annotations in time (i.e. regression) as well as session-level annotations (i.e. classification, or unique annotations for a given item).

# Installation
To install the latest version, use Julia 1.2 or greater. In a Julia REPL, do:
```julia
julia> ]
(v1.3) pkg> add www.github.com/usc-sail/AnnotationFusion.jl
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
We compute the pairwise distances between columns (considering the missing values) and mine triplets over these. The number of triplets mined depends on the number of items or sessions. By default, we mine `C*n*log(n)` triplets with `C = 40`. We then find an embedding representing the ratings using a logistic loss.

#### Copeland's method

### Regression (or real-time, or time-continuous annotations)

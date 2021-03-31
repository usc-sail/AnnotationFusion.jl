module AnnotationFusion

using SHA
using Random
using Distances
using DataFrames
using LinearAlgebra
using Statistics: mean, median
using TripletEmbeddings

import TripletEmbeddings: Triplets

export Triplets,
    dense, anonymize, anonymize!,
    Mean, Median, TE, Copeland,
    fuse, fuse!,
    name

# include("fusion.jl")
include("utils.jl")
include("triplets.jl")
include("methods.jl")

end
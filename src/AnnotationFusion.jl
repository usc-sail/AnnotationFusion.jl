module AnnotationFusion

using Random
using Distances
using DataFrames
using LinearAlgebra
using Statistics: mean, median
using TripletEmbeddings

import TripletEmbeddings: Triplets

export Triplets,
    dense,
    Mean, Median, TE,
    fuse, fuse!

# include("fusion.jl")
include("utils.jl")
include("triplets.jl")
include("methods.jl")

end
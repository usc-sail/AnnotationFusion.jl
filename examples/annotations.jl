using Plots
using Random
using DataFrames
using AnnotationFusion

Random.seed!(4)

n = 100
scores = rand(1:9, n)

annotations = DataFrame(items = 1:n, A = scores, B = round.(Int, clamp.(1.2 .* scores, 1, 9)))

μ = fuse(annotations, :items, Mean())

te = fuse(annotations, :items, TE())

plot(annotations.A); plot!(annotations.B); plot!(μ); plot!(te)
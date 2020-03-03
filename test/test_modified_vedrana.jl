using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
using Images
using Random
using ImageCore
using Distributions
using IterTools

function add_noise(sinogram)
    mx = maximum(sinogram)
    T = exp.(-sinogram/mx)*(10^3)
    noise = rand.(Poisson.(T))
    noisy = -log.(noise/10^3)
    return noisy # noisy sinogram
end

function pick_random(sinogram, nangles)
    num_bins, n_angles = size(sinogram)
    indexes = rand(1:n_angles, nangles)
    return indexes
end

function pick_best(sinogram, nangles, angles, w_angles=100.0, w_non=1.0)
    num_bins, n_angles = size(sinogram)
    best_count = 0.0
    best = collect(1:nangles)

    #determine how different angles should be
    tol = π/nangles -0.5
    for s in subsets(1:n_angles, nangles)
        a = angles[s]
        b = prepend!(deepcopy(a), 0.0)
        c = append!(deepcopy(a), π)
        differences = abs.(b-c)
        differences[1] = differences[1]+differences[end]
        pop!(differences)
        #Only take evenly distributed subsets
        if all(d -> d>tol,differences)
            #then pick those angles with most values
            non_zero_count = sum(x->x>0.0, sinogram[:,s])
            if non_zero_count> best_count
                best_count = non_zero_count
                best = s
            end
        end
    end
    return best
end

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "phantoms"))
cd(path)
#data = readdlm("hyperactive_sperm_trajectory.xyz", '\t')
data = readdlm("non_hyperactive_sperm_trajectory.xyz", '\t')
#remove first and last column which artime_sequence[:,1,1]e not relevant
data = data[1:end, 1:end .!= 1]
data = data[1:end, 1:end .!= 3]

#Remove rows that are not numeric
rows, columns = size(data)

numeric_rows = filter(i -> all(v->  isa(v, Number), data[i,:]), 1:rows)
index_diffs = (numeric_rows-circshift(numeric_rows,1))
data = data[numeric_rows, :]

# metrics of dataset
num_frames = 3000#1707
dims = 2
points = 38#40

#convert to time sequence (last dimension is frames)
time_sequence = zeros(points,dims,num_frames)
map(t -> time_sequence[1:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:num_frames)

#Radius function
r(s) = 1.0
bins = collect(-38.0:0.125:38.0)

frames = [1,2,3,4]
nangles = 5
sinograms = zeros(length(bins),nangles,length(frames))
ground_truth = Array[]
angles = zeros(nangles,length(frames))
for t in frames
    Random.seed!(t)
    #Remove all zero rows (missing data points)
    non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,1:points)

    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
    a = collect(0.0:0.1:π)#, 50)
    sinogram = parallel_forward(outline, a, bins)
    # subs = #filter!(s-> abs(a(s[1])-a(s[2])) > π/5 abs(a(s[1])-a(s[2]))&& subsets(1:length(a), nagles))
    # non_zero_count = sum(x->x>0.0, a, dims=1)
    best = pick_best(sinogram,nangles,a)
    angles[:,t] = a[best]#range(0.0,π, length=nangles)
    sinogram = sinogram[:,best]
    sinograms[:,:,t] = add_noise(sinogram)
    push!(ground_truth, outline)
end

tail_p = map(v -> 0.0-v*im, 0:1.0:38)
template = hcat(real(tail_p), imag(tail_p))
# f = scaleminmax(minimum(sinograms),maximum(sinograms))
# p2 = plot(Gray.(f.(sinograms[:,:,1])))
#p1 = plot(aspect_ratio=:equal)

plots = AbstractPlot[]
p = Progress(length(frames), 1)
for t in frames
    pl = plot(aspect_ratio=:equal)
    plot!(ground_truth[t][:,1], ground_truth[t][:,2], label="target", legend=:bottomleft)
    plot!(template[:,1], template[:,2], label="original")
    recon = recon2d_tail(deepcopy(template),r,angles[:,t],bins,sinograms[:,:,t],5000, 0.0, 1.0, 1)
    plot!(recon[:,1],recon[:,2], label ="result", framestyle=:none, legend=false)
    push!(plots,pl)
    next!(p)
end

path = normpath(joinpath(cwd, "results"))
l = @layout [a b c d e{0.1w}]

leg = plot(1:2, label="target")
plot!(1:2, label="original")
plot!(1:2, label="result", legend=true, xlim=(3,4), framestyle=:none)

plot(plots[1], plots[2], plots[3], plots[4], leg, layout=l, aspect_ratio=:equal, size=(1200,400))
cd(path)
savefig("reconstruction_vedrana.png")
cd(cwd)

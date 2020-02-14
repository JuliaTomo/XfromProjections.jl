using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
using Images

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
time_sequence = zeros(points+1,dims,num_frames)
map(t -> time_sequence[2:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:num_frames)

#Radius function
r(s) = 1.0



H,W = 381,381 #true size

nangles = 30
#angles = [0,π/2]#rand(0.0:0.001:π, nangles)
detcount = Int(floor(H*1.4))

halfdet = floor(detcount/2)
bins = collect(-halfdet:1.0:halfdet)

num_frames = 10
sinograms = zeros(detcount,nangles,num_frames)
ground_truth = Array[]
angles = zeros(nangles,num_frames)
for t in 1:num_frames
    #Remove all zero rows (missing data points)
    non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,2:points+1)
    prepend!(non_zeros,1)
    plot( aspect_ratio=:equal,framestyle=:none, legend=false)
    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
    angles[:,t] = rand(0.0:0.001:π, nangles)
    sinograms[:,:,t] = parallel_forward(outline, angles[:,t], bins)
    push!(ground_truth, outline)
end

tail_p = map(v -> 0.0-(v+0.0)*im, 0:2.5:18)
template = hcat(real(tail_p), imag(tail_p))

p1 = plot(aspect_ratio=:equal)
plot!(ground_truth[1][:,1], ground_truth[1][:,2], label="target", legend=:bottomleft)
plot!(template[:,1], template[:,2], label="original")


path = normpath(joinpath(cwd, "results"))
cd(path)

#
p1 = plot(aspect_ratio=:equal)
plot!(template[:,1], template[:,2], label="original")
p = Progress(num_frames, 1)
anim = @animate for t = 1:num_frames
    plot(deepcopy(p1))
    plot!(ground_truth[t][:,1], ground_truth[t][:,2], label="target", legend=:bottomleft)
    recon = recon2d_tail(deepcopy(template),r,angles[:,t],bins,sinograms[:,:,t],1000, 0.0, 0.1, 1)
    plot!(recon[:,1],recon[:,2], label ="result")
    next!(p)
end
gif(anim,"recon_tail.gif", fps=1)
cd(cwd)

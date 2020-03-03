using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
include("random_sperm.jl")

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
bins = collect(-38.0:0.125:38.0)

frames = [1,2]
possible_angles = collect(0.0:0.01:π)
angles = rand(possible_angles, length(frames))
nangles = 1
sinograms = zeros(length(bins),nangles,length(frames))
ground_truth = Array[]
for (i,t) in enumerate(frames)
    #Remove all zero rows (missing data points) (Except first head point)
    non_zeros = filter(j ->  any(v-> v !== 0.0, time_sequence[j,:,t]) ,1:points)
    prepend!(non_zeros,1)

    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
    sinogram = parallel_forward(outline, [angles[i]], bins)
    sinograms[:,:,i] = sinogram
    push!(ground_truth, reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)))
end


plot(ground_truth[1][:,1], ground_truth[1][:,2], aspect_ratio=:equal, label="frame 1")
plot!(ground_truth[2][:,1], ground_truth[2][:,2], aspect_ratio=:equal, label="frame 10")

known_sperm = ground_truth[2]
n = size(known_sperm, 1)

err = 10000
tol = 1.0e-2


max_iter = 1000
p = Progress(max_iter, 1)
iter = 1
best_recons, best_err, best_temp, best_sino, sino_err, best_sino_err = known_sperm, err, known_sperm, sinograms[:,1,2], 10000, 10000
while err > tol && iter <= max_iter
    #generate random feasible curve configuration
    template = iter == 1 ? known_sperm : generate_random_sperm(sinograms[:,1,1], known_sperm, angles[1], bins, n)
    outline, normals = get_outline(template, r)

    weights = ones(n)
    weights[1] = 0.0

    recon = recon2d_tail(deepcopy(template),r,[angles[1]],bins,sinograms[:,:,1],3000, 0.0, 0.1, 1, weights)

    outline, normals = get_outline(recon, r)
    fp = parallel_forward(outline, [angles[1]], bins)
    #global err = norm(sinograms[:,:,1]-fp)
    # global err = norm(ground_truth[1]-recon)
    # if err < best_err
    #     global best_recon = recon
    #     global best_err = err
    #     global best_temp = template
    # end

    global sino_err = norm(sinograms[:,:,1]-fp)
    if sino_err < best_sino_err
        global best_sino = recon
        global best_sino_err = sino_err
    end
    global iter +=1

    next!(p)
end

if err > tol
    @warn "Did not converge best error was %f" best_err
end

plot!(best_temp[:,1], best_temp[:,2], label="template")
plot!(best_recon[:,1], best_recon[:,2], label="reconstruction")

#Plot the detector and sinograms at the right angle to view difference related to the cells
#Consider mirroring if fit is not good
#consider how to use more angles

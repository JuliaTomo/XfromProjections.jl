using Logging
using Plots
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
using Printf
using LinearAlgebra
include("./sperm_phantom.jl")
include("./utils.jl")

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "phantoms"))
cd(path)
#data = readdlm("hyperactive_sperm_trajectory.xyz", '\t')
data = readdlm("non_hyperactive_sperm_trajectory.xyz", '\t')
#remove first and last column which are not relevant
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

#convert to time sequence (last dimension is frames) - add "head" at 0,0
time_sequence = zeros(points+1,dims,num_frames)
map(t -> time_sequence[2:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:num_frames)

#Radius function
r(s) = 1.0
bins = collect(-38.0:0.125:38.0)
cwd = @__DIR__
frame_nr = 1
nangles = 1
angles = [π/2]


#Remove all zero rows (missing data points)
non_zeros = filter(j ->  any(v-> v !== 0.0, time_sequence[j,:,frame_nr]) ,1:points)
prepend!(non_zeros,1)

#determine outline from skeleton
outline, normals = get_outline(reshape(time_sequence[non_zeros,:,frame_nr], (length(non_zeros),2)), r)
sinogram = parallel_forward(outline, angles, bins)
ground_truth = reshape(time_sequence[non_zeros,:,frame_nr], (length(non_zeros),2))

num_points = 30

template = get_straight_template(sinogram[:,1], r, [0.0 0.0], angles[1], num_points,bins)

plt = plot(template[:,1], template[:,2], aspect_ratio=:equal, label="template", framestyle=:none, color=:black, legend=false,  size=(400,400))
plot!(outline[:,1], outline[:,2], label = "target", color=:green)

n = 5000
recon1 = deepcopy(template)
recon2 = deepcopy(template)
w_u = ones(num_points+2)
w_u[1] = 0.0
w_u[2] = 0.0
w_l = ones(num_points+2)
w_l[1] = 0.0
w_l[2] = 0.0

recon1 = recon2d_tail(recon1,r,angles,bins,sinogram,999, 0.0, 0.1, 1, w_u, zeros(num_points+2))
recon2 = recon2d_tail(recon2,r,angles,bins,sinogram,999, 0.0, 0.1, 1, zeros(num_points+2), w_l)
recon1 = recon2d_tail(recon1,r,angles,bins,sinogram,1, 0.0, 0.1, 1, w_u, zeros(num_points+2), plot=true)
recon2 = recon2d_tail(recon2,r,angles,bins,sinogram,1, 0.0, 0.1, 1, zeros(num_points+2), w_l, plot=true)

plot!(recon1[:,1],recon1[:,2], label="upper", color=:red, linewidth=2)
plot!(recon2[:,1],recon2[:,2], label="lower", color=:blue, linewidth=2)

outline1, normals = get_outline(recon1, r)
s1 = parallel_forward(outline1, angles, bins)

outline2, normals = get_outline(recon2, r)
s2 = parallel_forward(outline2, angles, bins)

y_vals = cat(bins,s1,dims=2)
y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
plot_vals = y_vals[10:end-250,:]
plot!(plot_vals[:,1], plot_vals[:,2],label="FP upper", fill = (0, 0.2, :red), color=:white)

y_vals = cat(bins,s2,dims=2)
y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
plot_vals = y_vals[10:end-250,:]
plot!(plot_vals[:,1], plot_vals[:,2],label="FP lower", fill = (0, 0.2, :blue), color=:white)

y_vals = cat(bins,sinogram,dims=2)
y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
plot_vals = y_vals[10:end-250,:]
plot!(plot_vals[:,1], plot_vals[:,2],label="sinogram", color=:green)

path = normpath(joinpath(@__DIR__, "results/article_results"))
cd(path)
savefig("mirror")


cd(cwd)

plot!()

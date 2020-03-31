using DelimitedFiles
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
images, tracks = get_sperm_phantom(11,grid_size=0.1)

#Radius function
r(s) = 1.0
bins = collect(-38.0:0.125:38.0)
frame_nr = 1
nangles = 1
angles = [π/2]

#determine outline from skeleton
outline, normals = get_outline(tracks[1], r)
sinogram = parallel_forward(outline, angles, bins)

num_points = 30

template = get_straight_template(sinogram[:,1], r, [0.0 0.0], angles[1], num_points,bins)

plt = plot(template[:,1], template[:,2], aspect_ratio=:equal, label="template", framestyle=:none, color=:black,  size=(400,400), legend=false)
plot!(outline[:,1], outline[:,2], label = "target", color=:green)

n = 5000
recon = deepcopy(template)

w_u = ones(num_points+2)
w_u[1] = 0.0
w_u[2] = 0.0
w_l = ones(num_points+2)
w_l[1] = 0.0
w_l[2] = 0.0

recon = recon2d_tail(recon,r,angles,bins,sinogram,1, 0.0, 0.1, 1, w_u, w_l, plot=true)

plot!(recon[:,1],recon[:,2], label="reconstruction", color=:orange, linewidth=2)

outline, normals = get_outline(recon, r)
s = parallel_forward(outline, angles, bins)

y_vals = cat(bins,s,dims=2)
y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
plot_vals = y_vals[10:end-250,:]
plot!(plot_vals[:,1], plot_vals[:,2],label="FP", fill = (0, 0.2, :orange), color=:white)

y_vals = cat(bins,sinogram,dims=2)
y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
plot_vals = y_vals[10:end-250,:]
plot!(plot_vals[:,1], plot_vals[:,2],label="projection", color=:green)

path = normpath(joinpath(@__DIR__, "results/article_results"))
cd(path)
savefig("stalemate")

cd(cwd)

plot!()

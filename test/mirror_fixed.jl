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

#Radius function
r(s) = 1.0
bins = collect(-38.0:0.125:38.0)
cwd = @__DIR__
frame_nr = 1
nangles = 1
angles = [π/2]
ang = angles[1]

images, tracks = get_sperm_phantom(11,grid_size=0.1)
tail_length = curve_lengths(tracks[end])[end]

#determine outline from skeleton
outline, normals = get_outline(tracks[frame_nr], r)
sinogram = parallel_forward(outline, angles, bins)

num_points = 30

template = get_straight_template(sinogram[:,1], r, [0.0 0.0], angles[1], num_points,bins)

plt = plot(template[:,1], template[:,2], aspect_ratio=:equal, label="template", framestyle=:none, color=:black, legend=false)
plot!(outline[:,1], outline[:,2], label = "target", color=:green)


w_u = ones(num_points+2)
w_u[1] = 0.0
w_u[2] = 0.0
w_l = ones(num_points+2)
w_l[1] = 0.0
w_l[2] = 0.0

stepsize = 0.1
best_residual = Inf
best_recon = deepcopy(template)

recon1 = recon2d_tail(deepcopy(best_recon),r,angles,bins,sinogram,5000, 0.0, stepsize, 1, w_u, zeros(num_points+2))
recon2 = recon2d_tail(deepcopy(best_recon),r,angles,bins,sinogram,5000, 0.0, stepsize, 1, zeros(num_points+2), w_l)

initial1 = deepcopy(recon1)
initial2 = deepcopy(recon2)
for flip_pt=1:num_points
    recon1_flipped = flip(initial1,flip_pt,angles[1])
    recon2_flipped = flip(initial2,flip_pt,angles[1])
    #mirror and reconstruct with weights on both sides
    recon1 = recon2d_tail(deepcopy(recon1_flipped),r,angles,bins,sinogram,100, 0.0, stepsize, 1, w_u, zeros(num_points+2))
    recon2 = recon2d_tail(deepcopy(recon2_flipped),r,angles,bins,sinogram,100, 0.0, stepsize, 1, zeros(num_points+2), w_l)

    global best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, angles[1], bins, sinogram, best_recon, tail_length)
end

@info "keeping the best parts and restarting reconstruction"
recon_best = keep_best_parts(residual1, deepcopy(best_recon), ang, bins, 3, num_points, tail_length, sinogram[:,1], r)
recon1 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,sinogram,5000, 0.0, stepsize, 1, w_u, zeros(num_points+2))
recon2 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,sinogram,5000, 0.0, stepsize, 1, zeros(num_points+2), w_l)
best_residual,best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, sinogram, best_recon, tail_length)

initial1 = deepcopy(recon1)
initial2 = deepcopy(recon2)
for flip_pt=1:num_points
    recon1_flipped = flip(initial1,flip_pt,angles[1])
    recon2_flipped = flip(initial2,flip_pt,angles[1])
    #mirror and reconstruct with weights on both sides
    recon1 = recon2d_tail(deepcopy(recon1_flipped),r,angles,bins,sinogram,100, 0.0, stepsize, 1, w_u, zeros(num_points+2))
    recon2 = recon2d_tail(deepcopy(recon2_flipped),r,angles,bins,sinogram,100, 0.0, stepsize, 1, zeros(num_points+2), w_l)

    global best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, angles[1], bins, sinogram, best_recon, tail_length)
end

recon1 = flip(best_recon,1,ang)
recon2 = deepcopy(best_recon)
recon2d_tail(deepcopy(recon1),r,angles,bins,sinogram,1, 0.0, stepsize, 1, w_u, zeros(num_points+2), plot=true)
recon2d_tail(deepcopy(recon2),r,angles,bins,sinogram,1, 0.0, stepsize, 1, zeros(num_points+2), w_l, plot=true)
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
plot!(plot_vals[:,1], plot_vals[:,2],label="sinogram")

path = normpath(joinpath(@__DIR__, "results/article_results"))
cd(path)
savefig("mirror_fixed")


cd(cwd)

plot!()

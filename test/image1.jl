using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
# using Images
# using Random
# using ImageCore
# using Distributions
# using IterTools
# using Dierckx
using LinearAlgebra

function translate_points(list_points::Array{T,2}, translation::Array{T,2}) where T<:AbstractFloat
    return list_points.+translation
end

function rotate_points(list_points::Array{T,2}, ang::T) where T<:AbstractFloat
    rot = [cos(ang) sin(ang); -sin(ang) cos(ang)]
    return list_points*rot
end

function get_line(start_point, direction, n)
    return start_point.+direction.*(collect(0:n))
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

#convert to time sequence (last dimension is frames) - add "head" at 0,0
time_sequence = zeros(points+1,dims,num_frames)
map(t -> time_sequence[2:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:num_frames)

#Radius function
r(s) = 1.0
bins = collect(-38.0:0.125:38.0)

frame_nr = 1
nangles = 1
angles = [π/2]


#Remove all zero rows (missing data points)
non_zeros = filter(j ->  any(v-> v !== 0.0, time_sequence[j,:,frame_nr]) ,1:points)
prepend!(non_zeros,1)

#determine outline from skeleton
outline, normals = get_outline(reshape(time_sequence[non_zeros,:,1], (length(non_zeros),2)), r)
sinogram = parallel_forward(outline, angles, bins)
ground_truth = reshape(time_sequence[non_zeros,:,frame_nr], (length(non_zeros),2))

num_points = 30

template_y = range(0.0, -35.0, length=num_points)
template_x = zeros(num_points)
template = cat(template_x, template_y, dims = 2)

plt = plot(template[:,1], template[:,2], aspect_ratio=:equal, label="template")
plot!(outline[:,1], outline[:,2], label = "target")

x_line = cat(bins,zeros(length(bins)),dims=2)
y_vals = cat(bins,sinogram,dims=2)
x_line = rotate_points(translate_points(x_line, [0.0 20.0]), angles[1])
y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])



plot!(y_vals[10:end-250,1], y_vals[10:end-250,2], label="sinogram")
n = 4000
p = Progress(n, 1)
recon = deepcopy(template)

function find_largest_discrepancy(sino1,sino2)
    diff = abs.(sino1-sino2)
    i = argmax(diff)
    bin_max = bins[i]
    return bin_max
end

w_u = ones(num_points)
w_u[1] = 0.0
w_u[21:end] = zeros(10)
w_l = ones(num_points)
w_l[1:10] = zeros(10)
w_l[1] = 0.0
residual = 1000
anim1 = @animate for i=1:n
    plot(deepcopy(plt))
    if residual > 3
        global recon = recon2d_tail(recon,r,angles,bins,sinogram,1, 0.001, 0.1, 1, w_u, w_l)
    else
        global recon = recon2d_tail(recon,r,angles,bins,sinogram,1, 0.0, 0.1, 1, w_u, w_l)
    end
    plot!(recon[:,1],recon[:,2], label="result")
    outline, normals = get_outline(recon, r)
    s = parallel_forward(outline, angles, bins)
    global residual = norm(sinogram-s)
    y_vals = cat(bins,s,dims=2)
    y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
    plot!(y_vals[10:end-250,1], y_vals[10:end-250,2], label="recon_sino")
    next!(p)
end

# w_u = ones(num_points)
# w_u[1] = 0.0
# w_l = ones(num_points)
# w_l[1] = 0.0
#
# anim1 = @animate for i=1:n
#     plot(deepcopy(plt))
#     global recon = recon2d_tail(recon,r,angles,bins,sinogram,1, 0.001, 0.1, 1, w_u, w_l)
#     plot!(recon[:,1],recon[:,2], label="result")
#     outline, normals = get_outline(recon, r)
#     s = parallel_forward(outline, angles, bins)
#     y_vals = cat(bins,s,dims=2)
#     y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])
#     plot!(y_vals[10:end-250,1], y_vals[10:end-250,2], label="recon_sino")
#     next!(p)
# end

#flip


cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "results"))
cd(path)
gif(anim1, "evolution1.gif", fps = 40)
cd(cwd)

plot!()

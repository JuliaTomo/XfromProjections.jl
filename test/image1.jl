using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
using Printf
using LinearAlgebra
using Images
using Dierckx

function translate_points(list_points::Array{T,2}, translation::AbstractArray{T,2}) where T<:AbstractFloat
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
cwd = @__DIR__
p = Progress(10, 1)
L = curve_lengths(time_sequence[1:end,:,1])
for frame_nr = 1:10
#frame_nr = 2
    nangles = 1
    angles = [π/2]


    #Remove all zero rows (missing data points)
    non_zeros = filter(j ->  any(v-> v !== 0.0, time_sequence[j,:,frame_nr]) ,1:points)
    prepend!(non_zeros,1)
    L = curve_lengths(time_sequence[non_zeros,:,frame_nr])
    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,frame_nr], (length(non_zeros),2)), r)
    sinogram = parallel_forward(outline, angles, bins)
    ground_truth = reshape(time_sequence[non_zeros,:,frame_nr], (length(non_zeros),2))

    num_points = 30

    template_y = range(0.0, -35.0, length=num_points)
    template_x = zeros(num_points)
    template = cat(template_x, template_y, dims = 2)

    plt = plot(template[:,1], template[:,2], aspect_ratio=:equal, label="template", framestyle=:none, color=:black, legend=false, size=(400,400))
    plot!(outline[:,1], outline[:,2], label = "target", color=:green)

    x_line = cat(bins,zeros(length(bins)),dims=2)
    y_vals = cat(bins,sinogram,dims=2)
    x_line = rotate_points(translate_points(x_line, [0.0 20.0]), angles[1])
    y_vals = rotate_points(translate_points(y_vals, [0.0 20.0]), angles[1])


    n = 20000
    recon1 = deepcopy(template)
    recon2 = deepcopy(template)
    function find_largest_discrepancy(sino,current)
        residual = sino - current
        maxs = argmax(residual)
        bin_max = bins[maxs]
        return bin_max
    end

    w_u = ones(num_points+2)
    w_u[1] = 0.0
    w_u[2] = 0.0
    w_l = ones(num_points+2)
    w_l[1] = 0.0
    w_l[2] = 0.0
    #residual = 1000
    for i=1:n
        plot(deepcopy(plt))
        recon1 = recon2d_tail(recon1,r,angles,bins,sinogram,1, 0.0, 0.1, 1, w_u, zeros(num_points+2))
        recon2 = recon2d_tail(recon2,r,angles,bins,sinogram,1, 0.0, 0.1, 1, zeros(num_points+2), w_l)

        outline1, normals = get_outline(recon1, r)
        s1 = parallel_forward(outline1, angles, bins)
        outline2, normals = get_outline(recon2, r)
        s2 = parallel_forward(outline2, angles, bins)
        residual_mirror = norm(s2-s1)
        residual_1 = norm(sinogram-s1)
        residual_2 = norm(sinogram-s2)

        #if residual_mirror < 10e-2 && residual_1 > 1 && residual_2 > 1 && residual_1 < 6 && residual_2 < 6
        if i%1001 == 0
            b1 = find_largest_discrepancy(sinogram, s1)
            b2 = find_largest_discrepancy(sinogram, s2)
            #@info b1,b2
            angle = angles[1]
            projection = [cos(angle) sin(angle)]'
            vertex_coordinates1 = (recon1*projection)[:,1]
            vertex_coordinates2 = (recon2*projection)[:,1]

            v1 = findfirst(v -> v < b1, vertex_coordinates1)
            v2 = findfirst(v -> v < b2, vertex_coordinates2)

            if curve_lengths(recon1) > 1.5*L
                v1 = Int64(round(size(recon1,1)/2))
            end

            if curve_lengths(recon2) > 1.5*L
                v2 = Int64(round(size(recon2,1)/2))
            end

            if !isnothing(v1) && !isnothing(v2) && (v1 > 1 && v1 < num_points && v2 > 1 && v2 < num_points)
                needed_translation1 = (recon1[v1,:]-recon2[v1+1,:].+0.1)'
                needed_translation2 = (recon2[v2,:]-recon1[v2+1,:].+0.1)'
                temp = deepcopy(recon1)
                recon1 = cat(recon1[1:v1,:], translate_points(recon2[(v1+1):end,:],needed_translation1), dims=1)
                recon2 = cat(recon2[1:v2,:], translate_points(temp[(v2+1):end,:],needed_translation2), dims=1)
            end
        end
    end
    b1 = find_largest_discrepancy(sinogram, s1)
    b2 = find_largest_discrepancy(sinogram, s2)

    max1 = cat(b1,zeros(size(b1,1)), dims=2)
    max2 = cat(b2,zeros(size(b2,1)), dims=2)

    plot!(recon1[:,1],recon1[:,2], label="upper", color=:red)
    plot!(recon2[:,1],recon2[:,2], label="lower", color=:blue)

    outline1, normals = get_outline(recon1, r)
    s1 = parallel_forward(outline1, angles, bins)
    outline2, normals = get_outline(recon2, r)
    s2 = parallel_forward(outline2, angles, bins)

    y_vals1 = cat(bins,s1,dims=2)
    y_vals1 = rotate_points(translate_points(y_vals1, [0.0 20.0]), angles[1])
    plot!(y_vals1[10:end-250,1], y_vals1[10:end-250,2], label="upper sinogram", fill = (0, 0.2, :red), color=:white)
    y_vals2 = cat(bins,s2,dims=2)
    y_vals2 = rotate_points(translate_points(y_vals2, [0.0 20.0]), angles[1])
    plot!(y_vals2[10:end-250,1], y_vals2[10:end-250,2], label="lower sinogram", fill = (0, 0.2, :blue), color=:white)
    plot!(y_vals[10:end-250,1], y_vals[10:end-250,2], label="sinogram", color=:green)

    #maximums
    y_vals1 = rotate_points(translate_points(max1, [0.0 20.0]), angles[1])
    y_vals2 = rotate_points(translate_points(max2, [0.0 20.0]), angles[1])
    scatter!(y_vals1[:,1], y_vals1[:,2])
    scatter!(y_vals2[:,1], y_vals2[:,2])
    path = normpath(joinpath(@__DIR__, "results"))
    cd(path)
    savefig(@sprintf "frame_%d" frame_nr)#gif(anim1, "evolution1.gif", fps = 10)
    next!(p)
end

cd(cwd)

plot!()

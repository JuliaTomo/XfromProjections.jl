using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using Printf
using Measures
using Dierckx
using PolygonOps
include("../curves/curve_utils.jl")

function rotate_points(list_points::Array{T,2}, ang::T) where T<:AbstractFloat
    rot = [cos(ang) sin(ang); -sin(ang) cos(ang)]
    return list_points*rot
end

#use smaller grid than for reconstruction to avoid inverse crime
function get_sperm_phantom(nr_frames::Int64, r, grid)
    cwd = @__DIR__
    cd(cwd)
    #data = readdlm("hyperactive_sperm_trajectory.xyz", '\t')
    data = readdlm("trajectory.xyz", '\t')
    #remove first and last column which artime_sequence[:,1,1]e not relevant
    data = data[1:end, 1:end .!= 1]
    data = data[1:end, 1:end .!= 3]

    #Remove rows that are not numeric
    rows, columns = size(data)

    numeric_rows = filter(i -> all(v->  isa(v, Number), data[i,:]), 1:rows)
    data = data[numeric_rows, :]

    # metrics of dataset
    frames = 3000#1707
    dims = 2
    points = 38#40

    #convert to time sequence (last dimension is frames) - add "head" at 0,0
    time_sequence = zeros(points+1,dims,nr_frames)
    map(t -> time_sequence[2:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:nr_frames)

    p = Progress(5,1, "Making phantom")

    #Pick every 10th frame to match sampling at synkrotron
    images = zeros(length(grid),length(grid),nr_frames)
    tracks = Array{Float64,2}[]
    for t=1:nr_frames
        #Remove all zero rows (missing data points)
        non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,1:points)

        center_line = time_sequence[non_zeros,:,t]

        #find angles and rotatet the tail so it is on average parallel with the x-axis
        # this is done by converting points to complex numbers, finding the angles, taking average and turning -average
        cks = get_ck(center_line)

        xy = get_xy(cks)

        angles_est = angle.(cks[2:end])
        average_angle = sum(angles_est)/length(angles_est)
        center_line = rotate_points(xy, -average_angle)
        push!(tracks, center_line)

        #determine outline from skeleton
        outline, normals = get_outline(center_line, r)
        #close the curve
        outline = cat(outline, outline[1,:]', dims=1)
        #convert to binary image
        images[:,:,t] = closed_curve_to_binary_mat(outline,grid,grid)
        next!(p)
    end
    cd(cwd)
    return images, tracks
end

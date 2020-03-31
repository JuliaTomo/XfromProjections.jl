using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using Printf
using Measures
using Dierckx
using PolygonOps
using XfromProjections.curve_utils
using StaticArrays


#Makes a matrix where the matrix entry is true iff the center of corresponding pixel is not outside the closed curve
#curve is closed curve where first and last point should be the same.
#xs and ys denote the center of each pixel column/rownorm
function closed_curve_to_binary_mat(curve::Array{Float64}, xs::Array{Float64}, ys::Array{Float64})
    N = length(curve[:,1])
    poly = map(i -> SVector{2,Float64}(curve[i,1],curve[i,2]), 1:N)
    result = zeros(Bool, length(xs), length(ys))
    i = 0
    for x in xs
        i += 1
        j = 0#length(ys)+1
        for y in ys
            j +=1
            result[j,i] = inpolygon(SVector(x,y), poly) != 0 ? 1.0 : 0.0
        end
    end
    return result
end

#use smaller grid than for reconstruction to avoid inverse crime
function get_sperm_phantom(nr_frames::Int64; grid_size=0.1)
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
    data = data[numeric_rows, :]

    # metrics of dataset
    frames = 3000#1707
    dims = 2
    points = 38#40

    #convert to time sequence (last dimension is frames) - add "head" at 0,0
    time_sequence = zeros(points+1,dims,nr_frames)
    map(t -> time_sequence[2:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:nr_frames)

    p = Progress(5,1, "Making phantom")
    r(s) = 1.0
    #Pick every 10th frame to match sampling at synkrotron
    grid = collect(-38.0:grid_size:38.0)
    images = zeros(length(grid),length(grid),nr_frames)
    tracks = Array{Float64,2}[]
    for t=1:nr_frames
        #Remove all zero rows (missing data points)
        non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,1:points)
        prepend!(non_zeros,1)
        push!(tracks, (time_sequence[non_zeros,:,t]))

        #determine outline from skeleton
        outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
        #close the curve
        outline = cat(outline, outline[1,:]', dims=1)
        #convert to binary image
        images[:,:,t] = closed_curve_to_binary_mat(outline,grid,grid)
        next!(p)
    end
    cd(cwd)
    return images, tracks
end

H,W = 609, 609
function radon_operator(height, width, detcount, angle)
    angles = [angle]
    proj_geom = ProjGeom(0.5, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, height, width, -38.0,38.0, -38.0,38.0)
    return A
end

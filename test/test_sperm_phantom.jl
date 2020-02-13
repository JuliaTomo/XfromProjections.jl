using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using Dierckx
using PolygonOps
using StaticArrays

#### These should be moved to another file along with vedranas method
function segment_length(a::Array{Float64,1},b::Array{Float64,1})
    value = sqrt((a[1]-b[1])^2+(a[2]-b[2])^2)
    return value
end

function curve_lengths(arr::Array{Float64,2})
    result = Array{Float64,1}(undef,size(arr)[1])
    result[1]=0.0
    for i in 1:size(arr)[1]-1
        result[i+1] = segment_length(arr[i,:],arr[i+1,:])
    end
    csum = cumsum(result)
    return csum
end

function get_outline(centerline_points, radius_func)
    L = size(centerline_points,1)
    t = curve_lengths(centerline_points)
    spl = ParametricSpline(t,centerline_points',k=1)
    tspl = range(0, t[end], length=L)

    derr = derivative(spl,tspl)'
    normal = hcat(-derr[:,2], derr[:,1])
    radii = radius_func.(tspl)
    ronsplinetop = spl(tspl)'.+(radii.*normal)
    ronsplinebot = (spl(tspl)'.-(radii.*normal))[end:-1:1,:]

    outline_xy = cat(ronsplinetop, ronsplinebot, dims=1)
    return (outline_xy, normal)
end

#Makes a matrix where the matrix entry is true iff the center of corresponding pixel is not outside the closed curve
#curve is closed curve where first and last point should be the same.
#xs and ys denote the center of each pixel column/rownorm
function closed_curve_to_binary_mat(curve::Array{T1,2}, xs::Array{T2,1}, ys::Array{T2,1}) where {T1,T2 <: Number}
    N = length(curve[:,1])
    poly = map(i -> SVector{2,Float64}(curve[i,1],curve[i,2]), 1:N)
    result = zeros(Int64, length(xs), length(ys))
    i = 0
    for x in xs
        i += 1
        j = 0#length(ys)+1
        for y in ys
            j +=1
            result[j,i] = inpolygon(SVector(x,y), poly)
        end
    end
    return result
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
frames = 3000#1707
dims = 2
points = 38#40

#convert to time sequence (last dimension is frames)
time_sequence = zeros(points,dims,frames)
map(t -> time_sequence[:,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:frames)

p = Progress(length(1:frames),1, "Animating")
r(s) = 1.0
#Pick every 10th frame to match sampling at synkrotron
anim = @animate for t=1:frames
    #Remove all zero rows (missing data points)
    non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,1:points)
    plot( aspect_ratio=:equal,framestyle=:none, legend=false)
    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
    #close the curve
    outline = cat(outline, outline[1,:]', dims=1)
    #convert to binary image
    mat = closed_curve_to_binary_mat(outline,collect(-38.0:0.2:38.0),collect(-38.0:0.2:38.0))
    plot!(Gray.(mat))
    next!(p)
end

path = normpath(joinpath(@__DIR__, "results"))
cd(path)
#gif(anim, "hyper_active_sperm_phanton.gif", fps = 10) #500 fps is true frame rate
gif(anim, "non_hyper_active_sperm_phanton.gif", fps = 10) #500 fps is true frame rate


cd(cwd)

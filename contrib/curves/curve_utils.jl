#module curve_utils

#export snake_normals, eliminate_loopy_stuff, get_outline, closed_curve_to_binary_mat, curve_lengths

using Dierckx
using PolygonOps
using StaticArrays
using IterTools

function normalize!(n)
    lengths = map(i-> sum(n[i,:].^2)^0.5, 1:size(n)[1])
    lindexes = findall(l -> l > 0.0, lengths)
    n[lindexes,:] = n[lindexes,:]./lengths[lindexes]
end

function snake_normals(curve::AbstractArray{T,2}) where T<:AbstractFloat
    X = cat(dims = 1, curve[end,:]', curve, curve[1,:]'); # extended S
    dX = X[1:end-1,:]-X[2:end,:]; # dX
    normals_edges = hcat(dX[:,2], -dX[:,1])
    normalize!(normals_edges)
    normals_vertices = 0.5*(normals_edges[1:end-1,:]+normals_edges[2:end,:])
    normalize!(normals_vertices)
    return normals_vertices
end

function segment_length(a::AbstractArray{T,1},b::AbstractArray{T,1}) where T<:AbstractFloat
    value = sqrt((a[1]-b[1])^2+(a[2]-b[2])^2)
    return value
end

function curve_lengths(arr::AbstractArray{T,2}) where T<:AbstractFloat
    result = Array{Float64,1}(undef,size(arr)[1])
    result[1]=0.0
    for i in 1:size(arr)[1]-1
        result[i+1] = segment_length(arr[i,:],arr[i+1,:])
    end
    csum = cumsum(result)
    return csum
end

function eliminate_loopy_stuff(curve, limit)
    L = size(curve,1)
    tokeep = collect(1:L)
    for i in subsets(1:L, 2)
        #if the two points are not neighbours, and the distance between them is less than the limit and the loop is longer than 4*limit
        if abs(i[1]-i[2]) > 2 && segment_length(curve[i[1],:], curve[i[2],:]) < limit && curve_lengths(curve[i[1]:i[2],:])[end] > 4*limit#limit
            filter!(keeper -> keeper < (i[1]+1) || keeper > (i[2]-1), tokeep)
        end
    end
    curve = curve[tokeep,:]
    return curve
end

function get_xy(cks)
    Re = real(cks)
    Im = imag(cks)
    y = cumsum(Im);
    x = cumsum(Re);
    prepend!(y,[0.0]);
    prepend!(x,[0.0]);
    return cat(x,y, dims =2)
end

function get_ck(xy)
    xy_1 = xy[1:end-1,:]
    xy_2 = xy[2:end,:]
    vecs = xy_2.-xy_1
    return vecs[:,1]+im*vecs[:,2]
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

    head = (ronsplinetop[1,:].+ronsplinebot[end,:])./2
    tail = (ronsplinetop[end,:].+ronsplinebot[1,:])./2

    ronsplinetop = cat(head', ronsplinetop, dims = 1)
    ronsplinebot = cat(tail', ronsplinebot, dims = 1)

    outline_xy = cat(ronsplinetop, ronsplinebot, dims=1)

    normal = snake_normals(outline_xy)

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
        j = 0
        for y in ys
            j +=1
            result[j,i] = inpolygon(SVector(x,y), poly)
        end
    end
    return result
end

#end

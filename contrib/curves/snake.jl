#using PyCall
using Logging
using Dierckx
using IterTools
#using .snake_forward
#using .curve_utils
using Plots
using MATLAB
using ToeplitzMatrices

include("./snake_forward.jl")
include("./curve_utils.jl")
#Reimplementation of Vedranas method with modifications.

function regularization_matrix(N, alpha, beta)
    rr = zeros(N)
    rr[end-1:end] = alpha*[0 1] + beta*[-1 4]
    rr[1:3] = alpha*[-2 1 0] + beta*[-6 4 -1]
    return (I-Toeplitz(rr,rr))^-1
end


# #VEDRANA MODIFIED
# function displace(centerline_points, force, radius_func, w; doplot=false)
#     L = size(centerline_points,1)
#     (outline_xy, normal) = get_outline(centerline_points, radius_func)
#
#
#     #println(size(force), size(normal), size(w))
#
#     forces = (force.*normal).*w
#
#     displaced_outline = outline_xy .+ forces
#
#     mid = Int64(size(displaced_outline,1)/2)#always even
#     upper_points = displaced_outline[3:mid-1,:]
#     lower_points = (displaced_outline[mid+3:end-1, :])[end:-1:1,:]
#
#     displaced_centerline = (upper_points + lower_points)./2
#     head = (displaced_outline[1,:] + displaced_outline[2,:] + displaced_outline[end,:])./3
#     tail = (displaced_outline[mid,:] + displaced_outline[mid+1,:] + displaced_outline[mid+2,:])./3
#
#     displaced_centerline = cat(head', displaced_centerline, dims = 1)
#     displaced_centerline = cat(displaced_centerline, tail', dims = 1)
#
#     if doplot
#         plot!(outline_xy[:,1], outline_xy[:,2], label="original")
#         quiver!(outline_xy[:,1], outline_xy[:,2], quiver=(50*forces[:,1], 50*forces[:,2]), label = "forces")
#         plot!(displaced_outline[:,1], displaced_outline[:,2], label="displaced")
#         plot!(displaced_centerline[:,1], displaced_centerline[:,2], label="displaced")
#     end
#
#     return displaced_centerline
# end
#
# function move_points(residual,curves,angles,N,centerline_points,r,w; doplot=false)
#     (x_length, y_length) = size(residual)
#     vals = zeros(Float64, N)
#     if y_length > 1
#         F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
#         vals = zeros(Float64, N)
#         for i = 1:length(angles)
#             interp = F(curves[:,i], repeat([i], N))
#             vals += interp
#         end
#     else
#         F = Spline1D(collect(1:1.0:x_length), residual[:,1], k=1);
#         interp = F(curves[:,1])
#         vals = interp
#     end
#
#     force = vals*(1/length(angles))
#
#     centerline_points = displace(centerline_points, force, r, w, doplot=doplot)
#     return centerline_points
# end

function to_pixel_coordinates(current, angles, bins)
    N = size(current,1);
    vertex_coordinates = zeros(Float64,N,length(angles));
    a = (length(bins)-1)/(bins[end]-bins[1]); # slope
    b = 1-a*bins[1]; # intercept
    for k = 1:length(angles)
        angle = angles[k]
        projection = [cos(angle) sin(angle)]';
        #expressing vertex coordinates as coordinates in sinogram (pixel coordinates, not spatial)
        vertex_coordinates[:,k] = (current*projection)*a.+b;
    end
    return vertex_coordinates
end

# function evolve_curve(sinogram_target, centerline_points, r, angles, bins, max_iter, w, smoothness, degree::Int64; doplot=false)
#     (current, normal) = get_outline(centerline_points, r)
#     current_sinogram = parallel_forward(current,angles,bins)
#
#     curves = to_pixel_coordinates(current, angles, bins);
#     mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
#     residual = sinogram_target - mu*current_sinogram
#     N = size(current,1)
#     centerline_start = centerline_points[1,:]
#     for iter  = 1:max_iter
#         centerline_points = move_points(residual,curves,angles,N,centerline_points,r, w, doplot=doplot)
#
#         L = size(centerline_points,1)
#         #HACK
#         cp = eliminate_loopy_stuff(centerline_points, 2*r(0.0))
#         #HACK
#         if size(cp,1) > degree
#             centerline_points = cp
#         else
#             @warn "Too loopy"
#         end
#         t = curve_lengths(centerline_points)
#
#         spl = ParametricSpline(t,centerline_points',k=degree, s=0.0)
#         #HACK
#         if smoothness > 0.0
#             try
#                 spl = ParametricSpline(t,centerline_points',k=degree, s=smoothness)
#             catch e
#                 @warn e
#             end
#         end
#
#         tspl = range(0, t[end], length=L)
#         centerline_points = collect(spl(tspl)')
#         centerline_points[1,:] = centerline_start
#         (current, normal) = get_outline(centerline_points, r)
#         current_sinogram = parallel_forward(current,angles,bins);
#         curves = to_pixel_coordinates(current, angles, bins);
#         mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
#         residual = sinogram_target - mu*current_sinogram;
#     end
#     return centerline_points
# end
#
# function recon2d_tail(centerline_points::AbstractArray{T,2}, r, angles::Array{T},bins::Array{T},sinogram_target::Array{T,2}, max_iter::Int, smoothness::T, w::Array{T}, degree::Int64; doplot=false) where T<:AbstractFloat
#     current = evolve_curve(sinogram_target, centerline_points, r, angles, bins, max_iter, w, smoothness, degree, doplot=doplot)
#     return current
# end


# #VEDRANA ORIGINAL
# function regularization_matrix(N,alpha,beta)
#     cwd = @__DIR__
#     println(cwd)
#     mat"[$B, $A] = regularization_matrix($N,$alpha,$beta);"
#     return B
# end
#
function remove_crossings(curve)
    cwd = @__DIR__
    #println(cwd)
    mat"""
    addpath($cwd);
    $curve = remove_crossings($curve);"""
    return curve
end
#
function distribute_points(curve)
    curve = cat(dims = 1, curve, curve[1,:]'); # closing the curve
    N = size(curve,1); # number of points [+ 1, due to closing]
    dist = sqrt.(sum(diff(curve, dims=1).^2, dims=2))[:,1]; # edge segment lengths
    t = prepend!(cumsum(dist, dims=1)[:,1],0.0) # total curve length

    tq = range(0,t[end],length=N); # equidistant positions
    curve_new_1 = Spline1D(t,curve[:,1], k=1).(tq); # distributed x
    curve_new_2 = Spline1D(t,curve[:,2], k=1).(tq); # distributed y
    curve_new = hcat(curve_new_1,curve_new_2); # opening the curve again
    return curve_new[1:end-1,:]
end

function move_points(residual,curves,angles,N,centerline_points,r,w; doplot=false)
    (x_length, y_length) = size(residual)
    vals = zeros(Float64, N)
    if y_length > 1
        F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
        vals = zeros(Float64, N)
        for i = 1:length(angles)
            interp = F(curves[:,i], repeat([i], N))
            vals += interp
        end
    else
        F = Spline1D(collect(1:1.0:x_length), residual[:,1], k=1);
        interp = F(curves[:,1])
        vals = interp
    end

    force = vals*(1/length(angles))

    centerline_points = displace(centerline_points, force, r, w, doplot=doplot)
    return centerline_points
end

function move_points_original(residual,curves,angles,N,current,B,w)
    (x_length, y_length) = size(residual)
    #F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
    vals = zeros(Float64, N)
    if y_length > 1
        F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
        vals = zeros(Float64, N)
        for i = 1:length(angles)
            interp = F(curves[:,i], repeat([i], N))
            vals += interp
        end
    else
        F = Spline1D(collect(1:1.0:x_length), residual[:,1], k=1);
        interp = F(curves[:,1])
        vals = interp
    end
    force = vals*(1/length(angles))
    normals = snake_normals(current)
    vectors = force.*normals
    current = current + w*vectors;
    current = distribute_points(remove_crossings(B*current))
    return current
end

function evolve_curve_original(sinogram_target, current, angles, bins, B, max_iter, w,residuals,tol)
    current_sinogram = parallel_forward(current,angles,bins)

    curves = to_pixel_coordinates(current, angles, bins);

    mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
    residual = sinogram_target - mu*current_sinogram
    N = size(current,1)

    for iter  = 1:max_iter
        current = move_points_original(residual,curves,angles,N,current,B,w);
        current_sinogram = parallel_forward(current,angles,bins);
        curves = to_pixel_coordinates(current, angles, bins);
        mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
        residual = sinogram_target - mu*current_sinogram;
        error = norm(residual)/length(residual[:])
        append!( residuals, error )
        if error < tol
            @info "break at iteration " iter
            break
        end
    end
    return current, residuals
end

function vedrana(current::Array{T,2},angles::Array{T},bins::Array{T},sinogram_target::Array{T,2}, max_iter::Int, alpha::T, beta::T, w::T, tol) where T<:AbstractFloat
    N = size(current,1)
    B = regularization_matrix(N,alpha,beta)
    residuals = zeros(0)

    current, residuals = evolve_curve_original(sinogram_target, current, angles, bins, B, max_iter, w, residuals, tol)
    return current, residuals
end
#

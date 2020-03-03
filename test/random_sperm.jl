using Plots
using LinearAlgebra
using StatsBase
using Dierckx
using Colors
using ImageCore
using XfromProjections.curve_utils

function get_arc(center_point::Vector, radial_point::Vector, ang::Float64, n::Int64)
    radius = norm(radial_point-center_point)
    arc_length = radius*ang

    #Vector from center to radial
    center_radial = radial_point-center_point

    #Start arc at point -ang/2 from center_radial
    start_angle = angle(center_radial[1]+center_radial[2]*im)-ang/2

    #even angle steps
    θ = collect(LinRange(start_angle, start_angle+ang, n))

    #direction for each angle from center
    directions = cat(cos.(θ), sin.(θ), dims=2)
    return center_point'.+radius*directions
end

function get_weights(points::Array{T,2}, projection::Array{T,1}, bins::Array{T,1}, ang::T) where T <:  AbstractFloat
    projector = [cos(ang), sin(ang)]
    #Create a scale function that will scale values to have weights between 0 and 1 for whole sinogram
    scaler = scaleminmax(minimum(projection),maximum(projection))

    #Project to get "x" values to sinogram values
    weight_coordinates = (points*projector)

    #make spline for interpolating
    spl = Spline1D(bins,projection, k=1, bc="zero")
    weights = spl(weight_coordinates)

    #Scale weights
    return scaler.(weights)
end

function pick_random_point(points::Array{T,2}, weights::Array{T,1}) where T<:AbstractFloat
    l = size(points,1)
    points = map(i-> points[i,:], 1:l)
    return sample(points, Weights(weights))
end

#determines if point is to the right of line AB when standing in A looking at B
function is_in(A::Vector, B::Vector) where T<:AbstractFloat
     return f(p) = sign((B[1] - A[1])*(p[2] - A[2]) - (B[2] - A[2])*(p[1] - A[1])) <= 0.0
end

#function determining if point is in disc
function is_in(C::Vector, radius::Float64)
    return f(p) = (p[1]-C[1])^2 + (p[2]-C[2])^2 < radius^2
end

#NEEDS TESTING
function sperm_feasible_region(projection::Array{T,1}, previous_sperm::Array{T,2}, ang::T, bins::Array{T,1}) where T<:AbstractFloat
    L = curve_lengths(previous_sperm)[end]
    head = previous_sperm[end,:]
    neck = previous_sperm[end-1,:]
    d = neck - head
    perp = π/2
    rot = [cos(perp) sin(perp); -sin(perp) cos(perp)]
    #Line perpendicular to previous sperms neck (A,B)
    l1 = is_in(head, (d'*rot)')

    #Get left and right side - should account for noise soon
    b1 = findfirst(v -> v!= 0.0, projection)
    b2 = findlast(v -> v!= 0.0, projection)
    x_1= bins[b1]
    x_2= bins[b2]

    p_1 = [x_1, 1.0]
    p_2 = [x_2, 1.0]

    rot = [cos(ang) sin(ang); -sin(ang) cos(ang)]

    left_B = (p_1'*rot)'
    left_A = left_B+[sin(ang), -cos(ang)]

    right_A = (p_2'*rot)'
    right_B = right_A+[sin(ang), -cos(ang)]

    l2 = is_in(left_A, left_B)
    l3 = is_in(right_A, right_B)

    d = is_in(head, L)
    return f(p) = l1(p) && l2(p) && l3(p) && d(p)
end

function generate_random_sperm(projection::Array{T,1}, previous_sperm::Array{T,2}, ang::T, bins::Array{T,1}, n::Int64) where T <:AbstractFloat
    feasible_region = sperm_feasible_region(projection, previous_sperm, ang, bins)
    allowed_angle = π/3
    L = curve_lengths(previous_sperm)[end]
    seg_l = L/n

    head = previous_sperm[1,:]
    neck = previous_sperm[2,:]

    dir = neck - head
    radial = head + dir
    radial = (radial/norm(radial))*seg_l

    arc = get_arc(head, radial, allowed_angle, 10)
    weights = get_weights(arc, projection, bins, ang)
    p=pick_random_point(arc,weights)



    points = zeros(n,2)
    points[1,:] = head
    points[2,:] = p
    for i = 3:n
        prev_prev_point = points[i-2,:]
        prev_point = points[i-1,:]
        radial = prev_point+prev_point - prev_prev_point
        arc = get_arc(prev_point, radial, allowed_angle, 10)
        weights = get_weights(arc, projection, bins, ang)
        p=pick_random_point(arc,weights)
        maxiter, iter = 100, 1
        while !feasible_region(p) && iter < maxiter
            p=pick_random_point(arc,weights)
            iter += 1
        end
        points[i,:] = p
    end
    return points
end

# cp = [-0.0, -0.5]
# rp = [-0.6, -0.75]
# arc = get_arc(cp, rp, π/5, 100)
#
#
# ang = 0.0
# bins = collect(-1.0:0.1:1.0)
# sino = collect(1.0:-0.1:-1.0)
#
# w = get_weights(arc, sino, bins, ang)
#
# plot(arc[:,1], arc[:,2], aspect_ratio=:equal, lc=Gray.(w))
# scatter!([cp[1]], [cp[2]])
# scatter!([rp[1]], [rp[2]])

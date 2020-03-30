using Logging
using Dierckx
using LinearAlgebra
using StaticArrays
using XfromProjections
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using TomoForward
using ImageCore
using Suppressor
using Images
using LinearAlgebra
include("./sperm_phantom.jl")
include("./random_sperm.jl")

function find_largest_discrepancy(sino,current)
    residual = sino - current
    maxs = argmax(residual)
    bin_max = bins[maxs]
    return bin_max
end

function fix_length(centerline_points, tail_length, tol, L)
    #get "arclengths" for each point on the curve
    t = curve_lengths(centerline_points)
    #create spline
    spl = ParametricSpline(t,centerline_points', k=1, s=0.0)

    #If the tail has gotten too long then cut it
    if t[end] > tol*tail_length
        tspl = range(0, tail_length, length=L)
        centerline_points = spl(tspl)
    end

    return centerline_points
end

function get_worst_vertices(sinogram, ang, recon1, recon2, r)
    outline1, normals = get_outline(recon1, r)
    s1 = parallel_forward(outline1, [ang], bins)
    outline2, normals = get_outline(recon2, r)
    s2 = parallel_forward(outline2, [ang], bins)

    b1 = find_largest_discrepancy(sinogram, s1)
    b2 = find_largest_discrepancy(sinogram, s2)

    projection = [cos(ang) sin(ang)]'
    vertex_coordinates1 = (recon1*projection)[:,1]
    vertex_coordinates2 = (recon2*projection)[:,1]

    v1 = findfirst(v -> v < b1, vertex_coordinates1)
    v2 = findfirst(v -> v < b2, vertex_coordinates2)
    return v1, v2
end

function mirror_flip(sinogram, ang, recon1, recon2, r, num_points)
    v1, v2 = get_worst_vertices(sinogram,ang,recon1, recon2, r)

    if !isnothing(v1) && !isnothing(v2) && (v1 > 1 && v1 < num_points && v2 > 1 && v2 < num_points)
        needed_translation1 = (recon1[v1,:]-recon2[v1+1,:].+0.1)'
        needed_translation2 = (recon2[v2,:]-recon1[v2+1,:].+0.1)'
        temp = deepcopy(recon1)
        recon1 = cat(recon1[1:v1,:], translate_points(recon2[(v1+1):end,:],needed_translation1), dims=1)
        recon2 = cat(recon2[1:v2,:], translate_points(temp[(v2+1):end,:],needed_translation2), dims=1)

        #get arclengths
        t1 = curve_lengths(recon1)
        t2 = curve_lengths(recon2)
        spl1 = ParametricSpline(t1,recon1', k = 3, s=0.1)
        spl2 = ParametricSpline(t2,recon2', k = 3, s=0.1)
        recon1 = spl1(t1)
        recon2 = spl2(t2)
    end
    return recon1, recon2
end

function could_be_sperm_tail(tail_length, centerline_points)
    #Make a smoothed spline so we don't get changes from 'noise'
    L = size(centerline_points,1)
    t = curve_lengths(centerline_points)
    spl = ParametricSpline(t,centerline_points',k=2, s=3.0)
    tspl = range(0, t[end], length=L)
    smoothed = spl(tspl)'

    #plot!(smoothed[:,1], smoothed[:,2])

    prime = derivative(spl,tspl, nu=1)'
    primeprime = derivative(spl,tspl, nu=2)'

    k = (prime[:,1].*primeprime[:,2].-prime[:,2].*primeprime[:,1])./((prime[:,1].^2+prime[:,2].^2).^(3/2))
    cuvature_changes = count(x-> x!=0,(sign.(k)[2:end])-(circshift(sign.(k),1)[2:end]))

    #length check
    length_ok = t[end] <= tail_length+0.1*tail_length && t[end] >= tail_length-0.1*tail_length

    #tail is within +/- 10% estimated length, and curvature changes sign at most once.
    return cuvature_changes, k
end

function findMinAvgSubarray(arr, k)
    if length(arr) < k
        #TODO error here!
        return
    end

    result_index = 1;

    current_sum = sum(arr[1:k])

    minimum_sum = current_sum;

    for i = k+1:(length(arr)-k)
        current_sum = current_sum+arr[i] - arr[i-k]
        if current_sum < minimum_sum
            result_index = (i-k+1)
            minimum_sum = current_sum
        end
    end

    return result_index
end
#TODO add tail length and projection length to estimate the tail end point - two possibilities pick smoothest one
function best_parts(residual, centerline_points, ang, bins, k)
    first_third = Int64(round(size(centerline_points,1)/3))
    second_third = Int64(round(2*size(centerline_points,1)/3))

    head = centerline_points[1:first_third,:]
    mid = centerline_points[first_third+1:second_third,:]
    tail = centerline_points[second_third+1:end,:]

    magnitude = abs.(residual)

    F = Spline1D(bins, magnitude[:,1]);

    projection = [cos(ang) sin(ang)]'
    projected_head_part = (head*projection)[:,1]
    interp_head_residuals = F(projected_head_part)

    projected_mid_part = (mid*projection)[:,1]
    interp_mid_residuals = F(projected_mid_part)

    projected_tail_part = (tail*projection)[:,1]
    interp_tail_residuals = F(projected_tail_part)

    idx_best_head_part_start = findMinAvgSubarray(interp_head_residuals, k)
    idx_best_mid_part_start = findMinAvgSubarray(interp_mid_residuals, k)
    idx_best_tail_part_start = findMinAvgSubarray(interp_tail_residuals, k)
    head_indexes = unique(prepend!(collect(idx_best_head_part_start:idx_best_head_part_start+k-1),1))
    return cat(head[head_indexes,:], mid[idx_best_mid_part_start:idx_best_mid_part_start+k-1,:], tail[idx_best_tail_part_start:idx_best_tail_part_start+k-1,:], dims=1)
end

function estimate_projected_length(projection, ang, r,bins)
    projector = [cos(ang) sin(ang)]'

    #get the 'end points' of the projection, by getting the first and last value where value is greater than tail diameter, which is minimum
    projection_end1 = findfirst(p -> p > 2*r(0.0), projection)
    projection_end2 = findlast(p -> p > 2*r(0.0), projection)

    return abs(bins[projection_end1]-bins[projection_end2])
end
#(detector)   (a)
#|             /|
#|            / |
#|           /  |
#v         (b)_(c)
function keep_best_parts(residual, centerline_points, ang, bins, k, num_points, length, projection, r)
    best = best_parts(residual, centerline_points, ang, bins, k)

    estimated_length = curve_lengths(best)[end]
    ab = length - estimated_length

    fp = parallel_forward(get_outline(best, r)[1], [ang], bins)
    ac = estimate_projected_length(projection, ang, r, bins)-estimate_projected_length(fp[:,1], ang, r, bins)
    if ab > 0.0 && ac > 0.0
        a = best[end,:]
        a_prev = best[end-1,:]
        detector = [cos(ang), sin(ang)]
        ray = [sin(ang), -cos(ang)]
        cplus = a+detector*ac
        cminus = a-detector*ac
        c = norm(cplus-a_prev) > norm(cminus-a_prev) ? cplus : cminus
        bplus = c+ray*ab
        bminus = c-ray*ab
        b = norm(bplus-a_prev) > norm(bminus-a_prev) ? bplus : bminus
        best = cat(best,b', dims=1)
    end
    #create spline
    t = curve_lengths(best)
    spl = ParametricSpline(t,best', k=1, s=0.0)
    tspl = range(0, t[end], length=num_points)
    good_parts_of_tail = spl(tspl)'
    return good_parts_of_tail
end

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

detmin, detmax = -38.0, 38.0
function radon_operator(height, width, detcount, ang)
    proj_geom = ProjGeom(0.125, detcount, [ang])
    A = fp_op_parallel2d_line(proj_geom, height, width, detmin,detmax, detmin,detmax)
    return A
end

function get_straight_template(projection, r, head, ang, num_points,bins)
    projector = [cos(ang) sin(ang)]'
    projected_head = (head*projector)[1,1]

    #get the 'end points' of the projection, by getting the first and last value where value is greater than tail diameter, which is minimum
    projection_end1 = findfirst(p -> p > 2*r(0.0), projection)
    projection_end2 = findlast(p -> p > 2*r(0.0), projection)

    #determine the distance from head to each end point (in projection)
    dist_2_head1 = bins[projection_end1]-projected_head
    dist_2_head2 = bins[projection_end2]-projected_head

    #pick the largest distance
    projected_distance = abs(dist_2_head1) > abs(dist_2_head2) ? dist_2_head1 : dist_2_head2

    #Create line segment starting at head, going in direction of detector, and ending where the projection ends.
    return head.+projector'.*collect(range(0.0,projected_distance,length=num_points))
end

function reflection(v,l)
    return 2*(v*l./dot(l,l)).*repeat(l',size(v,1))-v
end

function try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
    residual1 = parallel_forward(get_outline(recon1, r)[1], [ang], bins) - projection
    residual2 = parallel_forward(get_outline(recon2, r)[1], [ang], bins) - projection
    ok1, k = could_be_sperm_tail(tail_length, recon1)
    ok2, k = could_be_sperm_tail(tail_length, recon2)
    if ok1 <= 1 && norm(residual1) < best_residual
        best_residual = norm(residual1)
        best_recon = recon1
    end

    if ok2 <= 1 && norm(residual2) < best_residual
        best_residual = norm(residual2)
        best_recon = recon2
    end

    return best_residual, best_recon, residual1, residual2
end

function flip(centerline_points, flip_point, ang)
    reflected = reflection(centerline_points[flip_point:end,:], [cos(ang), sin(ang)])
    first_part = centerline_points[1:flip_point-1,:]
    needed_translation = centerline_points[flip_point,:]-reflected[1,:]
    translated_and_reflected = reflected.+needed_translation'
    return cat(first_part, translated_and_reflected, dims=1)
end


using Plots
using Colors
cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "results/mirror_test/"))


images, tracks = get_sperm_phantom(21,grid_size=0.1)

detmin, detmax = -38.0, 38.0
grid = collect(detmin:0.1:detmax)
bins = collect(detmin:0.125:detmax)

ang = 2*π/3
angles, max_iter, stepsize = [ang], 10000, 0.1
tail_length = curve_lengths(tracks[end])[end]
num_points = 30
r(s) = 1.0
max_jiter = 1
frames2reconstruct = collect(1:10)
reconstructions = zeros(num_points,2,length(frames2reconstruct))
missed = zeros(num_points,2,length(frames2reconstruct))
while !isempty(frames2reconstruct)
    frame_nr = pop!(frames2reconstruct)
    jiter = 0

    #Get projection
    @info "making forward projection for frame: " frame_nr
    outline, normals = get_outline(tracks[frame_nr], r)
    projection = parallel_forward(outline, [ang], bins)

    #Add noise
    @info "adding gaussian noise at level 0.01"
    rho = 0.01
    e = randn(size(projection));
    e = rho*norm(projection)*e/norm(e);
    projection = projection + e;

    #Get ground truth of same length as recon
    t = curve_lengths(tracks[frame_nr])
    spl = ParametricSpline(t,tracks[frame_nr]', k=1, s=0.0)
    tspl = range(0, t[end], length=num_points)
    gt = spl(tspl)'

    angles = [ang]

    w_u = ones(num_points+2)
    w_u[1] = 0.0
    w_u[2] = 0.0
    w_l = ones(num_points+2)
    w_l[1] = 0.0
    w_l[2] = 0.0

    recon1_ok = false
    recon2_ok = false

    best_residual = Inf
    best_recon = zeros(num_points,2)
    while jiter < max_jiter # && best_residual > 1.5

        cd(savepath)
        jiter += 1
        @info jiter
        @info "setting up template"
        template = jiter==1 ? get_straight_template(projection[:,1], r, [0.0 0.0], ang, num_points,bins) : generate_random_sperm(projection[:,1], tracks[frame_nr+10], ang, bins, r, num_points)

        @info "calculating initial reconstruction"
        #Reconstruct with weights only on one side
        recon1 = recon2d_tail(deepcopy(template),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, w_u, zeros(num_points+2))
        recon2 = recon2d_tail(deepcopy(template),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, zeros(num_points+2), w_l)
        best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
        #plot
        # heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none)
        # plot!(template[:,1], template[:,2], aspect_ratio=:equal, label="template")
        # plot!(recon1[:,1], recon1[:,2], aspect_ratio=:equal, label=best_residual)
        # plot!(recon2[:,1], recon2[:,2], aspect_ratio=:equal, label=best_residual)
        # savefig(@sprintf "mirror_test_intermediate_%d" frame_nr)

        @info "checking if any parts could need mirroring"
        initial1 = deepcopy(recon1)
        initial2 = deepcopy(recon2)
        for flip_pt=1:num_points
            last_residual1, last_residual2 = residual1, residual2
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)
            #mirror and reconstruct with weights on both sides
            recon1 = recon2d_tail(deepcopy(recon1_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)
            recon2 = recon2d_tail(deepcopy(recon2_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)

            best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
            if norm(last_residual1) > norm(residual1) || norm(residual2) < norm(last_residual2)
                @info "IMPROVEMENT!"
            end

            #plot
            # heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none)
            # plot!(template[:,1], template[:,2], aspect_ratio=:equal, label="template")
            # plot!(recon1[:,1], recon1[:,2], aspect_ratio=:equal, label=norm(residual1))
            # plot!(recon2[:,1], recon2[:,2], aspect_ratio=:equal, label=norm(residual2))
            # plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, label=best_residual)
            # savefig(@sprintf "mirror_test_intermediate_%d_%d" frame_nr flip_pt)
        end

        @info "keeping the best parts and restarting reconstruction"
        recon_best = keep_best_parts(residual1, deepcopy(best_recon), ang, bins, 3, num_points, tail_length, projection[:,1], r)
        recon1 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, w_u, zeros(num_points+2))
        recon2 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, zeros(num_points+2), w_l)
        best_residual,best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)

        @info "checking if any parts could need mirroring 2"
        initial1 = deepcopy(recon1)
        initial2 = deepcopy(recon2)
        for flip_pt=1:num_points
            last_residual1, last_residual2 = residual1, residual2
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)
            #mirror and reconstruct with weights on both sides
            recon1 = recon2d_tail(deepcopy(recon1_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)
            recon2 = recon2d_tail(deepcopy(recon2_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)

            best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)

            #plot
            #heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none)
            # plot!(template[:,1], template[:,2], aspect_ratio=:equal, label="template")
            # plot!(recon1[:,1], recon1[:,2], aspect_ratio=:equal, label=norm(residual1))
            # plot!(recon2[:,1], recon2[:,2], aspect_ratio=:equal, label=norm(residual2))
            # plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, label=best_residual)
            # savefig(@sprintf "mirror_test_intermediate_%d_%d" frame_nr flip_pt)
        end
    end
    reconstructions[:,:,frame_nr] = best_recon
    @info "plotting"
    #Plot the ground truth
    heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none)
    #plot the best reconstruction
    plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, label=best_residual)
    #Plot the mirror
    mirror = flip(best_recon,1,ang)
    plot!(mirror[:,1], mirror[:,2], aspect_ratio=:equal, label="mirror")
    #save the figure
    savefig(@sprintf "mirror_test_%d" frame_nr)
end


cd(cwd)

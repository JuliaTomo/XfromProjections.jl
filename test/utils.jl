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
    return cuvature_changes, k, length_ok
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

    if projection_end1 == nothing || projection_end2 == nothing
        @warn "less than minimimum projection values"
        #TODO understand why!
        #just return some default value
        return 30
    else
        return abs(bins[projection_end1]-bins[projection_end2])
    end
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
    spl = ParametricSpline(t,best', k=1, s=1.0)
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
    ok1, k1, length_ok1 = could_be_sperm_tail(tail_length, recon1)
    ok2, k2, length_ok2 = could_be_sperm_tail(tail_length, recon2)

    if norm(residual1)+max(0,ok1-1) < best_residual  && length_ok1
        best_residual = norm(residual1)
        best_recon = recon1
    end

    if norm(residual2)+max(0,ok2-1) < best_residual  && length_ok2 # && ok2 <= 1
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

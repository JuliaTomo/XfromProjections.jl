#module snake_forward

#export parallel_forward

using Dierckx

function grouper(arr)
    arr = append!(arr,arr[1])
    result = Array[]
    group = []
    i = 1
    while i < length(arr)
        group = [i]
        i = i+1
        if  i <= length(arr) && arr[i-1] < arr[i]
            while i <= length(arr) && arr[i-1] < arr[i]
                if i == length(arr)
                    push!(group,1)
                else
                    push!(group,i)
                end
                i = i+1
            end
            push!(result, group)
            i = i-1
        elseif i <= length(arr) && arr[i-1] > arr[i]
            while i <= length(arr) && arr[i-1] > arr[i]
                if i == length(arr)
                    push!(group,1)
                else
                    push!(group,i)
                end
                i = i+1
            end
            push!(result, group)
            i = i-1
        end
    end
    if result[end][end] == result[1][1] && sign(result[end][end-1]-result[end][end]) == sign(result[1][1]-result[1][2])
        e = pop!(result)
        pop!(e)
        prepend!(result[1],e)
    end

    return result
end

function piece_wise_spline(splines, bins)
    values = zeros(Float64, length(bins))
    for i = 1:length(bins)
        b = bins[i]
        values[i] = sum(map(spl -> spl(b), splines))
    end
    return values
end

function project_curve(vertices, angle, bins)
    projection = [cos(angle) sin(angle)]'; # projection direction [along with detector]
    normal = [-projection[2] projection[1]]'; # normal direction [along with rays]

    # preparing for work
    vertex_coordinates = (vertices*projection)[:,1]; # vertex coordinates in projection
    distance = vertices*normal;# distances from vertex to projection
    groups = grouper(vertex_coordinates)
    projection = zeros(Float64, length(bins))
    splines = []
    for i = 1:length(groups)
        coordinates_ordered = vertex_coordinates[groups[i]]
        heights = distance[groups[i]]
        if coordinates_ordered[1] > coordinates_ordered[2]
            reverse!(coordinates_ordered)
            heights = -reverse(heights)
        end
        spl= Spline1D(coordinates_ordered,heights, k=1, bc="zero")
        push!(splines, spl)
    end
    return piece_wise_spline(splines, bins)
end

#forward model of list of vertices
function parallel_forward(vertices,angles,bins)
    N = size(vertices,1);
    indices = 1:N
    sinogram = zeros(Float64, length(bins),length(angles));
    vertex_coordinates = zeros(Float64,N,length(angles));
    position = zeros(Float64,N,1); # pre-allocating

    for k = 1:length(angles)

        angle = angles[k]
        projection = project_curve(vertices, angle, bins)

        sinogram[:,k] = projection
    end

    return sinogram
end

#end

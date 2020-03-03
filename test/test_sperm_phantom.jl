using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using Printf
using Measures

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

#convert to time sequence (last dimension is frames)
time_sequence = zeros(points,dims,frames)
map(t -> time_sequence[:,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:frames)

p = Progress(length(1:frames),1, "Animating")
r(s) = 1.0
#Pick every 10th frame to match sampling at synkrotron
path = normpath(joinpath(@__DIR__, "results"))
cd(path)
plots = AbstractPlot[]
for t=1:5
    #Remove all zero rows (missing data points)
    non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,1:points)
    pl = plot( aspect_ratio=:equal,framestyle=:none, legend=false)
    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
    #close the curve
    outline = cat(outline, outline[1,:]', dims=1)
    #convert to binary image
    mat = closed_curve_to_binary_mat(outline,collect(-38.0:0.125:38.0),collect(-38.0:0.125:38.0))
    plot!(Gray.(mat), title=@sprintf "t=%d" t)
    push!(plots, pl)
    savefig(@sprintf "Non-hyper_%d" t)
    next!(p)
end

l = @layout[a b c d e]
plot(plots..., layout = l, size=(1200,250), margin_left=0mm, margin_right=0mm)
# path = normpath(joinpath(@__DIR__, "results"))
# cd(path)
# #gif(anim, "hyper_active_sperm_phanton.gif", fps = 10) #500 fps is true frame rate
# gif(anim, "non_hyper_active_sperm_phanton.gif", fps = 10) #500 fps is true frame rate
savefig("Non-hyper")

cd(cwd)

using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
using Images
using Random
using ImageCore
using Distributions
using IterTools
using Dierckx
using LinearAlgebra

#tiny noise should calculate SNR...
function add_noise(sinogram)
    mx = maximum(sinogram)
    T = exp.(-sinogram/mx)*(10^5)
    noise = rand.(Poisson.(T))
    noisy = -log.(noise/10^5)
    return noisy # noisy sinogram
end

function translate_points(list_points::Array{T,2}, translation::Array{T,2}) where T<:AbstractFloat
    return list_points.+translation
end

function rotate_points(list_points::Array{T,2}, ang::T) where T<:AbstractFloat
    rot = [cos(ang) sin(ang); -sin(ang) cos(ang)]
    return list_points*rot
end

function get_line(start_point, direction, n)
    return start_point.+direction.*(collect(0:n))
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
num_frames = 3000#1707
dims = 2
points = 38#40

#convert to time sequence (last dimension is frames)
time_sequence = zeros(points,dims,num_frames)
map(t -> time_sequence[1:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:num_frames)

#Radius function
r(s) = 1.0
bins = collect(-38.0:0.125:38.0)

frames = [4]
nangles = 1
sinograms = zeros(length(bins),nangles,length(frames))
ground_truth = Array[]
full_sinos = Array[]
possible_angles = collect(0.0:0.01:π)
angles = rand(possible_angles, length(frames))#zeros(length(frames))##zeros(nangles,length(frames))#reshape([2.98,0.02,0.64,0.87,0.44,0.30,0.62],1,7)#reshape([2.9834,0.021,0.6403,0.8732,0.44449,0.3022,0.6167],1,7)#zeros(nangles,length(frames))
for (i,t) in enumerate(frames)
    #Remove all zero rows (missing data points)
    non_zeros = filter(j ->  any(v-> v !== 0.0, time_sequence[j,:,t]) ,1:points)

    #determine outline from skeleton
    outline, normals = get_outline(reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)), r)
    a = angles[i]
    sinogram = parallel_forward(outline, angles[:,i], bins)#sinogram[:,best]
    #add some random noise
    sinograms[:,:,i] = sinogram#add_noise(sinogram)
    push!(ground_truth, reshape(time_sequence[non_zeros,:,t], (length(non_zeros),2)))
end

# tail_p = map(v -> 0.0-v*im, 0:1.0:38)
# template = hcat(real(tail_p), imag(tail_p))
# # f = scaleminmax(minimum(sinograms),maximum(sinograms))
# # p2 = plot(Gray.(f.(sinograms[:,:,1])))
# #p1 = plot(aspect_ratio=:equal)

num_points = 30


plots = AbstractPlot[]
p = Progress(length(frames), 1)
recons = Array[]

function get_mid_and_tail(sinogram, ang, detector_distance, head; L = 20.0)
    b1 = findfirst(v -> v!= 0.0, sinogram)
    b2 = findlast(v -> v!= 0.0, sinogram)
    x_1= bins[b1]
    x_2= bins[b2]
    maxima = findlocalmaxima(sinogram)

    p = sortperm(sinogram[maxima])
    vals = sinogram[maxima]
    positions = bins[maxima]

    dist_x1 = abs(positions[1]-x_1)
    dist_x2 = abs(positions[1]-x_2)

    mid_x = dist_x1 > dist_x2 ? x_2 : x_1
    tail_x = x_1 == mid_x ? x_2 : x_1

    mid = rotate_points([mid_x detector_distance], ang)
    tail = rotate_points([tail_x detector_distance], ang)

    direction = [sin(ang) -cos(ang)]

    dist = detector_distance+norm(head*direction)#+L/2

    mid = mid + direction*dist
    tail = tail + direction*dist
    #dist_tail = detector_distance+norm(head*direction)#+L
    # mid_a = norm(head.-mid)
    # println(mid_a)
    # mid_c = L/2
    # mid_b = sqrt(mid_c^2-mid_a^2)
    # dist_mid = mid_b
    #
    # tail_a = norm(mid.-tail)
    # println(tail_a)
    # tail_c = L/2
    # tail_b = sqrt(tail_c^2-tail_a^2)
    # dist_tail = mid_b+tail_b

    mid = mid + direction*L/2
    tail = tail + direction*L

    return mid', tail'
end

w_u = ones(num_points)
w_u[1] = 0.0
w_l = ones(num_points)
w_l[1] = 0.0
for (i,t) in enumerate(frames)
    ang = angles[i]
    head = ground_truth[i][1,:]
    #perturb head slightly so that we don't assume to know it exactly
    head = head+[rand()*0.2 rand()*0.2]'

    #estimate mid and tail points
    sinogram = sinograms[:,1,i]
    detector_distance = bins[end]
    mid, tail = get_mid_and_tail(sinogram, ang, detector_distance, head)

    end_points = cat(head, mid, tail, dims=2)

    template = ParametricSpline(collect(range(1,num_points,length=size(end_points)[2])), end_points, k=1)(1:num_points)'

    outline, normals = get_outline(template, r)

    pl = plot(aspect_ratio=:equal)
    plot!(ground_truth[i][:,1], ground_truth[i][:,2], label="target", legend=:bottomleft)
    plot!(template[:,1], template[:,2], label="original")
    recon = recon2d_tail(deepcopy(template),r,angles[:,i],bins,sinograms[:,:,i],3000, 0.0, 0.1, 1, w_u, w_l)
    plot!(recon[:,1],recon[:,2], label ="result")
    push!(plots,pl)
    push!(recons,recon)
    next!(p)
end

path = normpath(joinpath(cwd, "results"))
l = @layout [a b c d e{0.1w}]

leg = plot(1:2, label="target")
plot!(1:2, label="original")
plot!(1:2, label="result", legend=true, xlim=(3,4), framestyle=:none)

#Plot the curves
#, plots[2], plots[3], plots[4], leg, layout=l, aspect_ratio=:equal, size=(1200,400))
# cd(path)
# savefig("reconstruction_vedrana.png")
# cd(cwd)

#Plot the sinograms
t = 1
plot(plots[t])
ang = angles[1]#rand(collect(0.0:0.01:π))

i = findfirst(v -> v ==ang, possible_angles)

#Get approximate x-value of end points
b1 = findfirst(v -> v!= 0.0, sinograms[:,1,t])
b2 = findlast(v -> v!= 0.0, sinograms[:,1,t])
x_1= bins[b1]
x_2= bins[b2]

x_line = cat(bins,zeros(length(bins)),dims=2)
y_vals = cat(bins,sinograms[:,1,t],dims=2)
x_line = rotate_points(translate_points(x_line, [0.0 bins[end]]), ang)
y_vals = rotate_points(translate_points(y_vals, [0.0 bins[end]]), ang)

# spl = Spline1D(bins,full_sinos[t][:,i], s=0.0, bc="zero", k=1)
#
# bmid = Int64(round((b2-b1)/2))

# smoothed = spl(bins[[b1,bmid,b2]])
plot!(x_line[:,1], x_line[:,2], label="detector")
plot!(y_vals[:,1], y_vals[:,2], label="sinogram")


# x=full_sinos[t][:,i]
# maxima = findlocalmaxima(x)
#
# p = sortperm(x[maxima])
# vals = x[maxima][p[end-1:end]]
# positions = bins[maxima][p[end-1:end]]
# xy_start = cat(positions,zeros(length(positions)), dims = 2)
# ang = π/2
# projection = [cos(ang) sin(ang)]
# start_points = xy_start.*projection
# println(xy_start, start_points)
#
# minima = findlocalminima(x)
# p = sortperm(x[minima])
# vals = x[minima][p[1:2]]
# positions = bins[minima][p[1:2]]
# # scatter!(bins[minima],x[minima])
# # scatter!(bins[maxima],x[maxima])
#
#
# # xs = bins
# # ys = spl(xs)
# # plot!(xs,ys)
# #
# recon = recons[t]
# outline, normals = get_outline(recon, r)
# a = collect(0.0:0.01:π)#, 50)
# result_sino = parallel_forward(outline, a, bins)[:,i]
# plot!(bins, result_sino)
#
b1 = findfirst(v -> v!= 0.0, sinograms[:,1,t])
b2 = findlast(v -> v!= 0.0, sinograms[:,1,t])
x_1= bins[b1]
x_2= bins[b2]
p1 = rotate_points(translate_points([x_1 0.0], [0.0 bins[end]]), ang)
p2 = rotate_points(translate_points([x_2 0.0], [0.0 bins[end]]), ang)
scatter!(p1[:,1],p1[:,2], label="p1")
scatter!(p2[:,1],p2[:,2], label="p2")
line_1 = get_line(p1, [sin(ang) -cos(ang)], 80)
line_2 = get_line(p2, [sin(ang) -cos(ang)], 80)
plot!(line_1[:,1], line_1[:,2], label="left")
plot!(line_2[:,1], line_2[:,2], label="right")

# x=sinograms[:,:,t]
# maxima = findlocalmaxima(x)
#
# p = sortperm(x[maxima])
# vals = x[maxima]#[p[end-1:end]]
# positions = bins[maxima]#[p[end-1:end]]
# x = cat(positions,zeros(length(positions)),dims=2)
# y = cat(positions,vals,dims=2)
# x = rotate_points(translate_points(x, [0.0 bins[end]]), ang)
# scatter!(x[:,1],x[:,2], label="maxima")
# line_maxima = get_line.(x, [sin(ang) -cos(ang)], 80)
# for i=1:size(line_maxima,1)
#     plot!(line_maxima[i,1],line_maxima[i,2],label="maxima line")
# end
# plot!()

# a, b, c, d = collect(range(x_1,x_2,length=4))
# b = a+3
# c = d-2
# int1 = integrate(spl, a,b)
# int2 = integrate(spl, b,c)
# int3 = integrate(spl, c,d)
# #plot!(bins,int)
# vals = [0.0,int1/(b-a), int2/(c-b), int3/(d-c)]
# plot!([a,b,c,d], vals, linetype=:steppre)
# val = integrate(spl, x_1, x_2)/(x_2-x_1)
# plot!([x_1,x_2],[0.0,val],linetype=:steppre)
# t = 3
# #find angle projecting onto x-axis (π/2)
# i = findfirst(v -> v ==0.0, collect(0.0:0.01:π))
#
# #Get approximate x-value of end points

#
# spl = Spline1D(bins,full_sinos[t][:,i], s=0.0, bc="zero", k=1)
#
# bmid = Int64(round((b2-b1)/2))
#
# # smoothed = spl(bins[[b1,bmid,b2]])
# plot(bins, full_sinos[t][:,i])
# plot!(bins, full_sinos[t+11][:,i])
# xs = bins
# ys = spl(xs)
# plot!(xs,ys)
#
# recon = recons[t]
# outline, normals = get_outline(recon, r)
# a = collect(0.0:0.01:π)#, 50)
# result_sino = parallel_forward(outline, a, bins)[:,i]
# plot!(bins, result_sino)
#
#
# # a, b, c, d = collect(range(x_1,x_2,length=4))
# # b = a+3
# # c = d-2
# # int1 = integrate(spl, a,b)
# # int2 = integrate(spl, b,c)
# # int3 = integrate(spl, c,d)
# # #plot!(bins,int)
# # vals = [0.0,int1/(b-a), int2/(c-b), int3/(d-c)]
# # plot!([a,b,c,d], vals, linetype=:steppre)
# val = integrate(spl, x_1, x_2)/(x_2-x_1)
# plot!([x_1,x_2],[0.0,val],linetype=:steppre)

# using Wavelets
# using LinearAlgebra
# b = collect(range(0.0,76.0,length=128))
# x = spl(b)#full_sinos[t][:,i]
#
# wt = wavelet(WT.haar)#, WT.Filter, WT.Periodic)
# #wt = wavelet(WT.cdf97, WT.Lifting)
# # 5 level transform
# xt = dwt(x, wt, 7)
#
# #xthresh = map(v -> v < 10.0 ? 0.0 : v, xt)
# # inverse tranform
# xti = idwt(xt, wt,7)
# # a full transform
# plot!(b,xti,label="inverse")

# # y = denoise(x,wt,L=6)    # regular dwt
# #
# # plot!(b,y,label="denoised")
# #
# # d,l = wplotdots(xt,0.1, 128)
#
# # scatter!(d,l)
# plot!(b,xt,label="transform",linetype=:steppost)

using SparseArrays
using PyCall
using LinearAlgebra
using TomoForward
using XfromProjections
include("../../phantoms/sperm_phantom.jl")

# H,W = 609, 609
# detmin, detmax = -38.0, 38.0
# function radon_operator(height, width, detcount, angles)
#     proj_geom = ProjGeom(0.5, detcount, angles)
#     A = fp_op_parallel2d_line(proj_geom, height, width, detmin,detmax, detmin,detmax)
#     return A
# end
# grid = collect(detmin:0.1:detmax)
# r(s) = 1.0
#
# images, tracks = get_sperm_phantom(301,r,grid)
# frames = images[:,:,collect(271:274)]
# @info size(frames)
#
# w_tv = 0.1
# w_flow  = 0.5
# c=1.1
#
#
# detcount = Int(floor(H*1.4))
#
# cwd = @__DIR__
# path = normpath(joinpath(@__DIR__, "result"))
# !isdir(path) && mkdir(path)
# cd(path)
#
# #4 angles evenly spaced between 0 and pi
# #angles = collect(range(0.0,π,length=5))[1:end-1]#angles = rand(0.0:0.001:π, 4)#collect(range(π/2,3*π/2,length=ang_nr+1))[1:end-1]
# #@info angles
# nangles = 4
# a = range(0.0,4*pi+pi/4, length=nangles*size(frames)[3])#rand(0.0:0.001:π, nangles*size(frames)[3])
# angles = map(t -> a[(t-1)*nangles+1:(t-1)*nangles+nangles], 1:size(frames)[3])
# @info angles
# #match size of input image (generating data)
# As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount, angles[t]),1:size(frames)[3])
# bs = zeros(size(As[1])[1],size(frames)[3])
# map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
#
# u0s = zeros(H,W,size(frames)[3])
# mask = ones(size(u0s)...)
#
# As  = map(t -> radon_operator(H,W, detcount, angles[t]),1:size(u0s)[3])
#
# @info "Reconstruction using tv regularization frame by frame"
# niter=450
# us_tv = zeros(H,W,size(frames)[3])
# for t = 1:size(frames)[3]
#    A = As[t]
#    p = bs[:,t]
#    u0 = u0s[:,:,t]
#    us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
# end

H, W = size(us_tv[:,:,1])

us = zeros(H,W,3)
us[:,:,1] = us_tv[:,:,1]
us[:,:,2] = us_tv[:,:,2]
us[:,:,3] = zeros(H,W)


vs = get_flows(us, 0.012)
v = vs[:,:,:,1]

Woptical = compute_warping_operator(v)
@views mul!(view(us[:,:,3], :), Woptical, vec(us[:,:,1]))

p1 = plot(Gray.(us[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="img1")
p2 = plot(Gray.(us[:,:,2]), aspect_ratio=:equal, framestyle=:none, title="img2")
p3 = plot(Gray.(us[:,:,3]), aspect_ratio=:equal, framestyle=:none, title="img1warped")
p4 = plot(Gray.(us[:,:,3]-us[:,:,2]), aspect_ratio=:equal, framestyle=:none, title="img1warped-img2")

l = @layout [a b c d]
plot(p1, p2, p3, p4, layout = l, size=(2000,600), linewidth=5)

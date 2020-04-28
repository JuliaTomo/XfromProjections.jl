using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
using PyCall
using Logging

include("./simple_phantoms.jl")
include("./tv_primaldual_flow.jl")
include("./optical_flow.jl")
include("./sperm_phantom.jl")

H,W = 609, 609
function radon_operator(height, width, detcount)
    nangles = 3
    angles = range(0.0,π, length=nangles+1)#rand(0.0:0.001:π, nangles)#
    proj_geom = ProjGeom(0.5, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, height, width, -38.0,38.0, -38.0,38.0)
    return A
end

#Create phantom at smaller gridsize than reconstruction
frames = get_sperm_phantom(10,grid_size=0.1)
detcount = Int(floor(H*1.4))
#match size of input image (generating data)
As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount),1:size(frames)[3])
bs = zeros(size(As[1])[1],size(frames)[3])
map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])

u0s = zeros(H,W,size(frames)[3])

As  = map(t -> radon_operator(H,W, detcount),1:size(u0s)[3])

w_tv = 0.3
w_flow  = 0.1
c=10.0


@info "Reconstruction using tv regularization frame by frame"
niter=450
us_tv = zeros(H,W,size(frames)[3])
for t = 1:size(frames)[3]
    A = As[t]
    p = bs[:,t]
    u0 = u0s[:,:,t]
    us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
end

@info "Reconstructing using joint motion estimation and reconstruction"
niter=50
u0s = deepcopy(us_tv)
us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, niter, niter, w_tv, w_flow, c)

@info "Preparing results in human readable format"
#run 450 iterations of the TV regularization first.
anim = @animate for t=1:size(frames)[3]
    l = @layout [a b c]
    p1 = plot(Gray.(frames[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="Ground truth", yflip=false)
    p2 = plot(Gray.(us_flow[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="Flow", yflip=false)
    p3 = plot(Gray.(us_tv[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="TV", yflip=false)
    plot(p1, p2, p3, layout = l)
end

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)
gif(anim, "reconstruction_flow.gif", fps = 1)
cd(cwd)

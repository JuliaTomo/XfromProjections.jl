using NPZ
using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
using PyCall
using Logging
include("../../phantoms/simple_phantoms.jl")
include("../tv_primaldual_flow.jl")
include("../optical_flow.jl")
include("../../../src/analytic/gridrec.jl")

cwd = @__DIR__
cd(cwd)

like(x::T, y) where T = convert(T, y)

proj = npzread("data/projections.npy")
theta = npzread("data/theta.npy")

nangles, imageZ, detcount = size(proj)

H, W = 100,100

function radon_operator(H,W, detcount, angles)
    proj_geom = ProjGeom(1.0, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, H, W)
    return A
end

pr = Float64.(reshape(sum(proj,dims=2), (nangles, detcount)))

window_step = 2
nframes = Int(nangles/window_step)

frames = zeros(H,W,nframes)
As = map(t -> radon_operator(H,W,detcount,theta[t:t+1]),1:window_step:nangles)
bs = reshape(pr,(detcount*window_step,nframes))
niter=500
u0s = zeros(H,W,nframes)

w_tv = 0.3
w_flow  = 0.05

@info "Reconstructing using joint motion estimation and reconstruction"
c=10.0
us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, 20, niter, w_tv, w_flow, c)

@info "Reconstruction using tv regularization frame by frame"
us_tv = zeros(H,W,nframes)
for t = 1:nframes
    A = As[t]
    p = bs[:,t]
    u0 = u0s[:,:,t]
    us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
end

@info "Preparing results in human readable format"
anim = @animate for t=1:nframes
    l = @layout [a b c]
    p1 = plot(Gray.(frames[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="Ground truth")
    p2 = plot(Gray.(us_flow[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="Flow")
    p3 = plot(Gray.(us_tv[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="TV")
    plot(p1, p2, p3, layout = l)
end

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)
gif(anim, "reconstruction_flow.gif", fps = 1)
cd(cwd)

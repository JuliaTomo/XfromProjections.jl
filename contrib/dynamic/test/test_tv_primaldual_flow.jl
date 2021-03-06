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

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)

#Translation
function transformation(ϕ)
    v1(x) = x
    v2(x) = x - ϕ*10

    t(x) = SArray{Tuple{2},Float64,1,2}(v1(x[1]),v2(x[2]))
    return t
end

H, W = 50,50#128, 128
img = zeros(H,W)
img = disc_phantom(img,15,25,8)
#img = imresize(img, H, W)
#p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Original")

function radon_operator(img)
    nangles = 2
    detcount = Int(floor(size(img,1)*1.4))
    angles = rand(0.0:0.001:π, nangles)
    proj_geom = ProjGeom(1.0, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
    return A
end

frames = zeros(H,W,20)
map(t -> frames[:,:,t+1]=replace_nan(warp(img, transformation(t*0.1), axes(img))), 0:size(frames)[3]-1)
As = map(t -> radon_operator(frames[:,:,t]),1:size(frames)[3])
bs = zeros(size(As[1])[1],size(frames)[3])
map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
niter=500
u0s = zeros(H,W,size(frames)[3])

w_tv = 0.3
w_flow  = 0.05

@info "Reconstructing using joint motion estimation and reconstruction"
c=10.0
us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, 20, niter, w_tv, w_flow, c)

@info "Reconstruction using tv regularization frame by frame"
us_tv = zeros(H,W,size(frames)[3])
for t = 1:size(frames)[3]
    A = As[t]
    p = bs[:,t]
    u0 = u0s[:,:,t]
    us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
end

@info "Preparing results in human readable format"
anim = @animate for t=1:size(frames)[3]
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

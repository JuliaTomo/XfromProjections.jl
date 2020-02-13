using TomoForward
using Images
using Plots
using XfromProjections
using ImageTransformations
using StaticArrays
using PyCall
using Logging
using LinearOperators

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)

#Define some non-linear transformation for warping
function nonlinear_transformation(ϕ)
    m1 = (sin(0.002*(ϕ*(257/(2*π))-1)*3/256))/4
    m2 = (sin(0.005*(ϕ*(257/(2*π))-1)*3/256))/4
    f(x) = x-128.5
    g1(x) = m1*x
    g2(x) = m2*x
    h(x) = x+1
    j(x) = x^5
    k(x) = x-1
    l1(x) = x/(5*m1)
    l2(x) = x/(5*m2)
    q(x) = x+128.5
    v1(x) = x |> f |> g1 |> h |> j |> k |> l1 |> q
    v2(x) = x |> f |> g2 |> h |> j |> k |> l2 |> q

    t(x) = SArray{Tuple{2},Float64,1,2}(v1(x[1]),v2(x[2]))
    return t
end

img = convert.(AbstractFloat,Gray.(load(normpath(joinpath(@__DIR__, "phantoms/shepplogan512.png")))))[:,:,1]

H, W = 128, 128
img = imresize(img, H, W)
p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Original")

function radon_operator(img)
    nangles = 10
    detcount = Int(floor(size(img,1)*1.4))
    angles = rand(0.0:0.001:π, nangles)
    proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])
    A = LinearOperator(fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2)))
    return A
end


frames = zeros(H,W,10)
map(t -> frames[:,:,t+1]=replace_nan(warp(img, nonlinear_transformation(t*0.1), axes(img))), 0:size(frames)[3]-1)
As = map(t -> radon_operator(frames[:,:,t]),1:size(frames)[3])
bs = zeros(size(As[1])[1],size(frames)[3])
map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
niter=800
u0s = zeros(H,W,size(frames)[3])

w_tv = 0.001
w_flow  = 0.1

@info "Reconstructing using joint motion estimation and reconstruction"
us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, niter, niter, w_tv, w_flow)

@info "Reconstruction using tv regularization frame by frame"
us_tv = zeros(H,W,size(frames)[3])
for t = 1:size(frames)[3]
    A = As[t]
    p = bs[:,t]
    u0 = u0s[:,:,t]
    us_tv[:,:,t] = recon2d_tv_primaldual(A, p, u0, niter, w_tv)
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
path = normpath(joinpath(@__DIR__, "results"))
cd(path)
gif(anim, "reconstruction.gif", fps = 1)
cd(cwd)

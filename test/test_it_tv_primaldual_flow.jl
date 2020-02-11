using Revise
using TomoForward
using SparseArrays
using Images
using Plots
using XfromProjections
using ImageTransformations
using StaticArrays
using PyCall
using Interpolations

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)

#Define some non-linear transformation and the inverse
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

img = convert.(AbstractFloat,Gray.(load(normpath(joinpath(@__DIR__, "test_data/shepplogan512.png")))))[:,:,1]

H, W = 128, 128
img = imresize(img, H, W)
p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Original")

# trans =nonlinear_transformation(0.5)
# imgw = replace_nan(warp(img, trans, axes(img)))
# p_img_w = plot(Gray.(imgw), aspect_ratio=:equal, framestyle=:none, title="Image Warped")

# test line projection model
function radon_operator(img)
    nangles = 10
    detcount = Int(floor(size(img,1)*1.4))

    proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])
    A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
    return A
end

frames=map(deg -> replace_nan(warp(img, nonlinear_transformation(deg), axes(img))), 0:0.1:1.0)
As = map(f -> radon_operator(f),frames)
ps = map(t -> As[t]*vec(frames[t]), 1:length(frames))
niter=800
u0s = zeros(H,W,length(frames))
us = recon2d_tv_primaldual_flow(As, ps, u0s, niter, 0.01, 0.5)

p_1 = plot(Gray.(us[:,:,8]), aspect_ratio=:equal, framestyle=:none, title="recon")

#l = @layout [a b; c]
#plot(p_img, p_1, p_img_w, layout=l)

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

# py"""
# import numpy as np
# import pyflow
#
# def flow(img1,img2,alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=1, nInnerFPIterations=1, nSORIterations=30, colType=1):
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     height, width = img1.shape
#
#     im1 = np.array(img1).reshape(height,width,1).copy(order='C')
#     im2 = np.array(img2).reshape(height,width,1).copy(order='C')
#
#     u, v, im2W = pyflow.coarse2fine_flow(
#         im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
#         nSORIterations, colType)
#
#     return u, v, im2W
# """

function imWarp(img, dx, dy)
    itp = interpolate(img, BSpline(Linear()))
    H,W = size(img)
    imw = zeros(H,W)
    for I in CartesianIndices(img)
        dxi, dyi = dx[I], dy[I]
        y, x = clamp(I[1] + dyi, 1, H), clamp(I[2] + dxi, 1, W)
        imw[I] = itp[y, x]
    end
    return imw
end

img = convert.(AbstractFloat,Gray.(load(normpath(joinpath(@__DIR__, "test_data/shepplogan512.png")))))[:,:,1]

H, W = 128, 128
img = imresize(img, H, W)
p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Original")

trans =nonlinear_transformation(0.8)
imgw = replace_nan(warp(img, trans, axes(img)))
p_img_w = plot(Gray.(imgw), aspect_ratio=:equal, framestyle=:none, title="Image Warped")

# flow_x, flow_y, im2Warped = py"flow"(img,imgw)
# p_warped_gt = plot(Gray.(im2Warped[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="Image Warped back")
#
# flow = cat(flow_x,flow_y, dims=3)
# Warper = compute_warping_operator(flow)
# im_vec = collect(Iterators.flatten(img))
#
# im_warp_op = reshape(Warper*im_vec, H, W)
# p_op = plot(Gray.(im_warp_op), aspect_ratio=:equal, framestyle=:none, title="Warping operator")
#
# # im_warp_py = reshape(warper_py*im_vec, H, W)
# # p_py = plot(Gray.(im_warp_py), aspect_ratio=:equal, framestyle=:none, title="Warping py")
#
# l = @layout [a b; c d]
#
# # img_w_intp = imWarp(imgw, flow_x, flow_y)
# # p_intp = plot(Gray.(img_w_intp), aspect_ratio=:equal, framestyle=:none, title="Interpolator")
# plot(p_img, p_img_w, p_warped_gt, p_op, layout = l)

img = zeros(128, 128)
img[40:60, 40:60] .= 1.0

nangles = 30
detcount = Int(floor(size(img,1)*1.4))
####A1########
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

# test line projection model
A_1 = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
p_1 = A_1 * vec(img)
###############

p_2 = A_1 * vec(imgw)


# p_noise = p + rand(size(p, 1))
# p = reshape(Array(p), (:, detcount));

u0 = zeros(size(img))
niter=800
lambdas = [0.01, 0.1, 0.6]
As = [A_1, A_1]
u0s = cat(u0, u0, dims = 3)
bs = cat(p_1, p_2, dims = 2)
us = recon2d_tv_primaldual_flow(As, bs, u0s, niter, 0.01, 0.5)

p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Image")
p_1 = plot(Gray.(us[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[3]) c=0.01")

l = @layout [a b; c]
plot(p_img, p_1, p_img_w, layout=l)

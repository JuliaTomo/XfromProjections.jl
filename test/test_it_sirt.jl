using Revise
using TomoForward
using XfromProjections
using SparseArrays
using Images
using PyPlot

img = imread("test_data/shepplogan512.png")[:,:,1]

H, W = 256, 256
img = imresize(img, H, W)

# img = zeros(128, 128)
# img[40:60, 40:60] .= 1.0

nangles = 360
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
p = A * vec(img)

u0 = zeros(size(img))
niter=300
@time u = recon2d_sirt(A, p, u0, niter; min_const=0.0)

imshow(u, cmap="gray")
title("reconstructed image")
show()
"""Compute A' * Filter( A( LoG * f ) ) from Projections

where
LoG : Laplacian of Gaussian filter
f : object
A : Radon transform

Ref:

[1] 1992_Srinivasa,Ramakrishnan,Rajgopal_Detection_of_edges_from_projections IEEE_Transactions_on_Medical_Imaging
"""

using TomoForward
using XfromProjections
using SparseArrays
using PyPlot

include("./edge_from_proj.jl")
# img = imread("test_data/shepplogan512.png")[:,:,1]

img = zeros(100, 100)
img[20:50, 30:40] .= 1

nangles = 100
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
@time p = A * vec(img);
p = reshape(Array(p), (:, detcount));
q = filter_proj(p)

p_log = radon_log(q, 1.0)

@time LoG = A' * vec(p_log)
fbp = A' * vec(q)

LoG_img = reshape(LoG, size(img))
fbp_img = reshape(fbp, size(img))

ax00 = plt.subplot2grid((2,3), (0,0))
ax01 = plt.subplot2grid((2,3), (0,1))
ax02 = plt.subplot2grid((2,3), (0,2))
ax10 = plt.subplot2grid((2,3), (1,0))
ax11 = plt.subplot2grid((2,3), (1,1))
ax12 = plt.subplot2grid((2,3), (1,2))
ax00.imshow(p'); ax00.set_title("sinogram p")
ax01.imshow(fbp_img); ax01.set_title("fbp")
# ax02.imshow(bp_img_strip); ax02.set_title("bp_strip")
ax10.imshow(p_log); ax10.set_title("p * R(LoG)")
ax11.imshow(LoG_img, cmap="coolwarm"); ax11.set_title("LoG")
# ax12.imshow(fbp_img_strip); ax12.set_title("fbp_strip")
plt.show()

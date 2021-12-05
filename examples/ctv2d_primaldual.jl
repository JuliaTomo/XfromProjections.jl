using TomoForward
using XfromProjections
using LinearAlgebra

# img = imread("test_data/shepplogan512.png")[:,:,1]
# H, W = 128, 128
# img = imresize(img, H, W)

img1 = zeros(128, 128)
img1[40:60, 40:60] .= 1.0
H, W = size(img1)

img2 = zeros(H, W)
img2[40:60, 40:60] .= 2.0

src_origin = 876.25
det_origin = 352.2525

nangles = 128
angles = deg2rad.(LinRange(0, 360, nangles+1))[1:nangles]


detcount = Int(floor(size(img1,1)*1.4))
proj_geom = ProjGeomFan(0.8, detcount, angles, src_origin, det_origin)

# test line projection model
A = fp_op_fan_line(proj_geom, size(img1, 1), size(img1, 2))
p1_ = A * vec(img1)
p1 = reshape(p1_, nangles, detcount)

p2_ = A * vec(img2)
p2 = reshape(p2_, nangles, detcount)

nchannels = 2
u = zeros(H, W, nchannels)
p = cat(p1, p2, dims=3)

niter=500
w_tnv=1.0

reg_type = "l∞11" # or "tnv"
res = recon2d_ctv_primaldual!(u, A, p, niter, 1.0001, reg_type, 1e-6, 1, 0.5)
u = res[1]
res_primals = res[3]
res_duals = res[4]
residuals = res[3] + res[4]

using Plots
plot(res_primals, yaxis=:log)
plot!(res_duals)
plot!(residuals)

# plot reconstruction image for the first channel
plot(Gray.(u[:,:,1]))

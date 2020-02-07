using Revise
using TomoForward
using SparseArrays
using Images
using Plots
using XfromProjections

img = convert.(AbstractFloat,Gray.(load(normpath(joinpath(@__DIR__, "test_data/shepplogan512.png")))))[:,:,1]

H, W = 128, 128
img = imresize(img, H, W)

# img = zeros(128, 128)
# img[40:60, 40:60] .= 1.0

nangles = 30
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

# test line projection model
A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
p = A * vec(img)
# p_noise = p + rand(size(p, 1))
# p = reshape(Array(p), (:, detcount));

u0 = zeros(size(img))
niter=800
lambdas = [0.01, 0.1, 0.6]
As = [A]
u0s = reshape(u0, (size(u0)[1], size(u0)[2],1))
bs = reshape(p, (length(p),1))
us = recon2d_tv_primaldual_flow(As, bs, u0s, niter, 0.01, 0.5)

p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Image")
p_1 = plot(Gray.(us[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[3]) c=0.01")

l = @layout [a b]
plot(p_img, p_1, layout=l)

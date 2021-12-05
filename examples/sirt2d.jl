using TomoForward
using XfromProjections

img = zeros(128, 128)
img[40:60, 40:60] .= 1.0
H, W = size(img)

nangles = 90
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
p = reshape(A * vec(img), nangles, detcount)

u = zeros(size(img))
niter=100
@time recon2d_sirt!(u, A, p, niter)

# using PyPlot
# imshow(u, cmap="gray")
# title("reconstructed image")
# show()
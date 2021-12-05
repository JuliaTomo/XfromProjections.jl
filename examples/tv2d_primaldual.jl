using TomoForward
using XfromProjections

# img = imread("test_data/shepplogan512.png")[:,:,1]
# H, W = 128, 128
# img = imresize(img, H, W)

img = zeros(128, 128)
img[40:60, 40:60] .= 1.0
H, W = size(img)

nangles = 30
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

# test line projection model
A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
p = A * vec(img)
p = reshape(p, nangles, detcount)

niter=500
lambda = 0.01
u0 = zeros(size(img))
recon2d_tv_primaldual!(u0, A, p, niter, lambda)

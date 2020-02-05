using Revise
using TomoForward
using XfromProjections
using SparseArrays
using Images
using PyPlot

img = imread("test_data/shepplogan512.png")[:,:,1]

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
u = zeros(H, W, 5)
u[:,:,1] = recon2d_tv_primaldual(A, p, u0, niter, lambdas[3], 0.01)
u[:,:,2] = recon2d_tv_primaldual(A, p, u0, niter, lambdas[3], 10)

for (i,lamb) in enumerate(lambdas)
    w_tv=lamb
    c=1.0

    u[:,:,i+2] = recon2d_tv_primaldual(A, p, u0, niter, w_tv, c)
end

ax00 = plt.subplot2grid((2,3), (0,0))
ax01 = plt.subplot2grid((2,3), (0,1))
ax02 = plt.subplot2grid((2,3), (0,2))
ax10 = plt.subplot2grid((2,3), (1,0))
ax11 = plt.subplot2grid((2,3), (1,1))
ax12 = plt.subplot2grid((2,3), (1,2))
ax00.imshow(img); ax00.set_title("original")
ax01.imshow(u[:,:,1]); ax01.set_title("lamb $(lambdas[3]) c=0.01")
ax02.imshow(u[:,:,2]); ax02.set_title("lamb $(lambdas[3]) c=10")
ax10.imshow(u[:,:,3]); ax10.set_title("lamb $(lambdas[1]) c=1")
ax11.imshow(u[:,:,4]); ax11.set_title("lamb $(lambdas[2]) c=1")
ax12.imshow(u[:,:,5]); ax12.set_title("lamb $(lambdas[3]) c=1")
suptitle("original and reconstruction results")
show()
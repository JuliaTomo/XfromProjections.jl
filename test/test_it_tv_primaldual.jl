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
u = zeros(H, W, 5)
u[:,:,1] = recon2d_tv_primaldual(A, p, u0, niter, lambdas[3], 0.01)
u[:,:,2] = recon2d_tv_primaldual(A, p, u0, niter, lambdas[3], 10)

for (i,lamb) in enumerate(lambdas)
    w_tv=lamb
    c=1.0

    u[:,:,i+2] = recon2d_tv_primaldual(A, p, u0, niter, w_tv, c)
end

p_img = plot(Gray.(img), aspect_ratio=:equal, framestyle=:none, title="Image")
p_1 = plot(Gray.(u[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[3]) c=0.01")
p_2 = plot(Gray.(u[:,:,2]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[3]) c=10")
p_3 = plot(Gray.(u[:,:,3]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[1]) c=1")
p_4 = plot(Gray.(u[:,:,4]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[2]) c=1")
p_5 = plot(Gray.(u[:,:,5]), aspect_ratio=:equal, framestyle=:none, title="lamb $(lambdas[3]) c=1")

l = @layout [a b c; c d e]
plot(p_img, p_1, p_2, p_3, p_4, p_5, layout=l)

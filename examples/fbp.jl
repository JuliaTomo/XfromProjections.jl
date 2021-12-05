using TomoForward
using XfromProjections

img = zeros(100, 100)
img[30:51, 40:81] .= 1

nangles = 90
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

# test line projection model
A = fp_op_parallel2d_line(proj_geom, size(img, 1), size(img, 2))
@time p = A * vec(img);
p = reshape(Array(p), (:, detcount));
q = filter_proj(p)

@time bp = A' * vec(p)
fbp = A' * vec(q) .* (pi / nangles)

fbp_img = reshape(fbp, size(img))
bp_img = reshape(bp, size(img))

# test strip projection model
A_strip = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
@time p = A_strip * vec(img);
p = reshape(Array(p), (:, detcount));
q = filter_proj(p)

p_ = vec(p)
q_ = vec(q)

bp_strip = A' * p_
@time fbp_strip = A' * q_ .* (pi / nangles)

fbp_img_strip = reshape(fbp_strip, size(img))
bp_img_strip = reshape(bp_strip, size(img))

using PyPlot

ax00 = plt.subplot2grid((2,3), (0,0))
ax01 = plt.subplot2grid((2,3), (0,1))
ax02 = plt.subplot2grid((2,3), (0,2))
ax10 = plt.subplot2grid((2,3), (1,0))
ax11 = plt.subplot2grid((2,3), (1,1))
ax12 = plt.subplot2grid((2,3), (1,2))
ax00.imshow(p); ax00.set_title("sinogram")
ax01.imshow(bp_img); ax01.set_title("bp")
ax02.imshow(bp_img_strip); ax02.set_title("bp_strip")
ax10.imshow(img); ax10.set_title("ground truth")
ax11.imshow(fbp_img); ax11.set_title("fbp")
ax12.imshow(fbp_img_strip); ax12.set_title("fbp_strip")
plt.show()

# make a synthetic projection
incldue("../src/analytic/gridrec.jl")
using TomoForward

img = zeros(128, 128, 2)
img[10:80, 20:110, 1:2] .= 1.0

nslice = size(img, 3)

nangles = 90
detcount = 196
# detcount = Int(floor(size(img,1)*1.4))
angles = Array(LinRange(0,pi,nangles+1)[1:nangles])
proj_geom = ProjGeom(1.0, detcount, angles)

isdefined_A = @isdefined A
if isdefined_A == false
    A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
end

p = zeros(nangles, nslice, detcount)
for i=1:nslice
    p[:,i,:] = reshape(A * vec(img[:,:,i]), nangles, detcount)
end

@time rec_img = recon2d_gridrec(p, angles)

using PyPlot
imshow(rec_img[:,:,1])
# imshow(img[:,:,1])
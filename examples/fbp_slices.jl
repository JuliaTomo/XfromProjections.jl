using TomoForward
using XfromProjections

# using MKLSparse # uncomment if you've installed MKLSparse, which will boost the performance

# generate synthetic data
img = zeros(128, 128, 128)
img[40:70, 40:60, 50:70] .= 1.0
H, W, nslice = size(img)

nangles = 90
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

isdefined_A = @isdefined A
if isdefined_A == false
    A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
end

p = zeros(nangles, nslice, detcount)
for i=1:nslice
    p[:,i,:] .= reshape(A * vec(img[:,:,i]), nangles, detcount)
end

# fbp slice by slice
q = filter_proj(p)
img3d = bp_slices(q, A, H, W)

# using PyPlot
# imshow(img3d[:,:,60])
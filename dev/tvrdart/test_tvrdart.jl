using TomoForward
using XfromProjections

# using MKLSparse # uncomment if you've installed MKLSparse, which will boost the performance

# test slice by slice
img = zeros(100, 100, 128)
img[40:70, 40:60, 50:70] .= 1.0

nslice = size(img, 3)

nangles = 90
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

isdefined_A = @isdefined A
if isdefined_A == false
    A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
end

p = zeros(nangles, nslice, detcount)
for i=1:nslice
    p[:,i,:] = reshape(A * vec(img[:,:,i]), nangles, detcount)
end

u = zeros(size(img))

include("pals.jl")
recon2d_slices_tvrdart!(u, A, p, 50, 4)


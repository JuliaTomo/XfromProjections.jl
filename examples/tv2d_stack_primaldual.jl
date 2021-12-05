using TomoForward
using XfromProjections

# using MKLSparse # uncomment if you've installed MKLSparse, it will boost the performance

# test slice by slice
img = zeros(128, 128, 128)
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
    p[:,i,:] = reshape(A * vec(img[:,:,i]), nangles, detcount);
end

p = permutedims(p, [2,3,1])

u0 = zeros(size(img));
@time u = recon2d_stack_tv_primaldual!(u0, A, p, 20, 0.1);

# using PyPlot
# imshow(u[:,:,60])


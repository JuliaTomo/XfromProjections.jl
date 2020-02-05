using XfromProjections
using FileIO
using MeshIO

mesh = load("../test/test_data/bunny.obj")

# H, W = 256, 362
H, W = 50, 50 # detector size
nangles = 2
angles = LinRange(0, pi, nangles)[1:end-1]
proj_geom = ProjGeom(3.0/W, 3.0/H, H, W, angles)

sinogram = fp_mesh(proj_geom, mesh)

using PyPlot
idx=1
imshow(sinogram[idx,:,:])
show()
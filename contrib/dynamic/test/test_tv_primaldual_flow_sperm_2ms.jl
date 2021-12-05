using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
using PyCall
using Logging
using Random
using Printf

#include("../tv_primaldual_flow.jl")
#include("../optical_flow.jl")
include("../../phantoms/sperm_phantom.jl")

H,W = 609, 609
detmin, detmax = -38.0, 38.0
function radon_operator(height, width, detcount, angles)
    proj_geom = ProjGeom(0.5, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, height, width, detmin,detmax, detmin,detmax)
    return A
end
grid = collect(detmin:0.1:detmax)
r(s) = 1.0

images, tracks = get_sperm_phantom(301,r,grid)
frames = images[:,:,collect(1:31)]
@info size(frames)

w_tv = 0.1
w_flow  = 0.05
c=1.1


detcount = Int(floor(H*1.4))

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)

#4 angles evenly spaced between 0 and pi
#angles = collect(range(0.0,π,length=5))[1:end-1]#angles = rand(0.0:0.001:π, 4)#collect(range(π/2,3*π/2,length=ang_nr+1))[1:end-1]
#@info angles
nangles = 3
a =rand(0.0:0.001:π, nangles*size(frames)[3]) #range(0.0,4*pi+pi/4, length=nangles*size(frames)[3])#
angles = map(t -> a[(t-1)*nangles+1:(t-1)*nangles+nangles], 1:size(frames)[3])
@info angles
#match size of input image (generating data)
As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount, angles[t]),1:size(frames)[3])
bsex = zeros(size(As[1])[1],size(frames)[3])

map(t -> bsex[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
@info "adding gaussion noise at level 0.01"
rho = 0.01
e = randn(size(bsex));
e = rho*norm(bsex)*e/norm(e);
bs = bsex + e;

u0s = zeros(H,W,size(frames)[3])
mask = ones(size(u0s)...)

As  = map(t -> radon_operator(H,W, detcount, angles[t]),1:size(u0s)[3])

@info "Reconstruction using tv regularization frame by frame"
niter=450
us_tv = zeros(H,W,size(frames)[3])
for t = 1:size(frames)[3]
    A = As[t]
    p = bs[:,t]
    u0 = u0s[:,:,t]
    us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
end
p1 = heatmap(grid, grid, Gray.(us_tv[:,:,1]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[271][:,1], tracks[271][:,2], aspect_ratio=:equal, linewidth=5)
p2 = heatmap(grid, grid, Gray.(us_tv[:,:,2]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[272][:,1], tracks[272][:,2], aspect_ratio=:equal, linewidth=5)
p3 = heatmap(grid, grid, Gray.(us_tv[:,:,3]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[273][:,1], tracks[273][:,2], aspect_ratio=:equal, linewidth=5)
p4 = heatmap(grid, grid, Gray.(us_tv[:,:,4]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[274][:,1], tracks[274][:,2], aspect_ratio=:equal, linewidth=5)
l = @layout [a b c d]
plot(p1, p2, p3, p4, layout = l, size=(2000,600), linewidth=5)
savefig(@sprintf "result_all_tv_2_%d" length(angles))

@info "Reconstructing using joint motion estimation and reconstruction"
niter1=100
niter2=500
u0s = deepcopy(us_tv)
us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, niter1, niter2, w_tv, w_flow, c, mask, 0.012)

p1 = heatmap(grid, grid, Gray.(us_flow[:,:,1]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[271][:,1], tracks[271][:,2], aspect_ratio=:equal, linewidth=5)
p2 = heatmap(grid, grid, Gray.(us_flow[:,:,2]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[272][:,1], tracks[272][:,2], aspect_ratio=:equal, linewidth=5)
p3 = heatmap(grid, grid, Gray.(us_flow[:,:,3]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[273][:,1], tracks[273][:,2], aspect_ratio=:equal, linewidth=5)
p4 = heatmap(grid, grid, Gray.(us_flow[:,:,4]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
plot!(tracks[274][:,1], tracks[274][:,2], aspect_ratio=:equal, linewidth=5)
l = @layout [a b c d]
plot(p1, p2, p3, p4, layout = l, size=(2000,600), linewidth=5)
savefig(@sprintf "result_all_flow_2_%d" length(angles))

@info "Preparing results in human readable format"
anim = @animate for t=1:size(frames)[3]
    l = @layout [a b c]
    p1 = heatmap(grid, grid, Gray.(frames[:,:,t]),yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false, title="Ground Truth")
    p2 = heatmap(grid, grid, Gray.(us_flow[:,:,t]),yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false, title="Flow")
    plot!(tracks[t+270][:,1], tracks[t+270][:,2], aspect_ratio=:equal, linewidth=1)
    p3 = heatmap(grid, grid, Gray.(us_tv[:,:,t]),yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false, title="TV")
    plot!(tracks[t+270][:,1], tracks[t+270][:,2], aspect_ratio=:equal, linewidth=1)
    plot(p1, p2, p3, layout = l)
end


cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)
gif(anim, "reconstruction_flow_tv_2.gif", fps = 1)
cd(cwd)

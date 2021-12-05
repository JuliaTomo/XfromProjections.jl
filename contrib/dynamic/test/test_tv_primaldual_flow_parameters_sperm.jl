using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
#using PyCall
using Logging
using Printf
using Suppressor

#include("../../phantoms/simple_phantoms.jl")
#include("../tv_primaldual_flow.jl")
#include("../optical_flow.jl")
include("../../phantoms/sperm_phantom.jl")

H,W = 609, 609
detmin, detmax = -38.0, 38.0
function radon_operator(height, width, detcount, angles)
    proj_geom = ProjGeom(0.5, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, height, width, -38.0,38.0, -38.0,38.0)
    return A
end

#Create phantom at smaller gridsize than reconstruction
grid = collect(detmin:0.1:detmax)
r(s) = 1.0
frames, tracks = get_sperm_phantom(2,r,grid)
detcount = Int(floor(H*1.4))
#match size of input image (generating data)
nangles = 4
a = range(0.0,2*pi+pi/4, length=nangles*size(frames)[3])#rand(0.0:0.001:π, nangles*size(frames)[3])
angles = map(t -> a[(t-1)*nangles+1:(t-1)*nangles+nangles], 1:size(frames)[3])
As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount, angles[t]),1:size(frames)[3])

bsex = zeros(size(As[1])[1],size(frames)[3])

map(t -> bsex[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
@info "adding gaussion noise at level 0.01"
rho = 0.01
e = randn(size(bsex));
e = rho*norm(bsex)*e/norm(e);
bs = bsex + e;

u0s = zeros(H,W,size(frames)[3])

As  = map(t -> radon_operator(H,W, detcount, angles[t]),1:size(u0s)[3])
cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)

u0s = zeros(H,W,size(frames)[3])
mask = ones(size(u0s)...)

c=1.1
w_tvs = [0.0001,0.001,0.01,0.05,0.1,0.2,0.3,0.5]
w_flows = [0.0001,0.001,0.01,0.05,0.1,0.2,0.3,0.5]
prog = Progress(size(w_tvs,1)*size(w_flows,1),60, "Testing parameters")

for w_tv = w_tvs
    niter=450
    us_tv = zeros(H,W,size(frames)[3])
    for t = 1:size(frames)[3]
        A = As[t]
        p = bs[:,t]
        u0 = u0s[:,:,t]
        @suppress begin
            us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
        end
    end
    for w_flow = w_flows
        niter=50
        @suppress begin
            us_flow = recon2d_tv_primaldual_flow(As, bs, deepcopy(us_tv), niter, 200, w_tv, w_flow, c, mask, 0.012)
            filenamepng = @sprintf "flow_2ms_%f_%f__3.png" w_tv w_flow;
            #PLOT
            plot(Gray.(us_flow[:,:,1]), aspect_ratio=:equal, framestyle=:none, legend=false, yflip=false)
            plot!(tracks[1][:,1], tracks[1][:,2], aspect_ratio=:equal, linewidth=5)
            savefig(filenamepng)
            next!(prog)
        end
    end
end

# angles = rand(1:0.001:π,3)
# #Create phantom at smaller gridsize than reconstruction
# frames, tracks = get_sperm_phantom(2,r, grid)
# detcount = Int(floor(H*1.4))
# #match size of input image (generating data)
# As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount, angles[t]),1:size(frames)[3])
# bsex = zeros(size(As[1])[1],size(frames)[3])
#
# map(t -> bsex[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
# @info "adding gaussion noise at level 0.01"
# rho = 0.01
# e = randn(size(bsex));
# e = rho*norm(bsex)*e/norm(e);
# bs = bsex + e;
#
# As  = map(t -> radon_operator(H,W, detcount, angles[t]),1:size(u0s)[3])
#
# for w_tv = w_tvs
#     niter=450
#     us_tv = zeros(H,W,size(frames)[3])
#     for t = 1:size(frames)[3]
#         A = As[t]
#         p = bs[:,t]
#         u0 = u0s[:,:,t]
#         @suppress begin
#             us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
#         end
#     end
#     for w_flow = w_flows
#         niter=50
#         @suppress begin
#             us_flow = recon2d_tv_primaldual_flow(As, bs, deepcopy(us_tv), niter, 20, w_tv, w_flow, c)
#             filenamepng = @sprintf "flow_radn_ang_2ms_%f_%f_.png" w_tv w_flow;
#             #PLOT
#             plot(Gray.(us_flow[:,:,1]), aspect_ratio=:equal, framestyle=:none, legend=false, yflip=false)
#             savefig(filenamepng)
#             next!(p)
#         end
#     end
# end
# cd(cwd)

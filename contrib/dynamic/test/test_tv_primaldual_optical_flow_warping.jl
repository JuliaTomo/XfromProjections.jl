using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
using PyCall
using Logging

include("../tv_primaldual_flow.jl")
include("../optical_flow.jl")
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
frames = images[:,:,collect(271:274)]
@info size(frames)

w_tv = 0.01
w_flow  = 0.001
c=10.0


detcount = Int(floor(H*1.4))

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)

#4 angles evenly spaced between 0 and pi
angles = [rand(0.0:0.001:π, 3), rand(0.0:0.001:π, 3), rand(0.0:0.001:π, 3), rand(0.0:0.001:π, 3)]#collect(range(π/2,3*π/2,length=ang_nr+1))[1:end-1]
@info angles
#match size of input image (generating data)
As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount, angles[t]),1:size(frames)[3])
bs = zeros(size(As[1])[1],size(frames)[3])
map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])

u0s = zeros(H,W,size(frames)[3])

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

img2 = us_tv[:,:,2]#zeros(128,128)
img1 = us_tv[:,:,1]
H, W = size(img2)
#img2[30:60,40:50] .= 1.0

#flow = zeros(H, W, 2)
#flow[:,:, 1] .= -13.0
#flow[:,:, 2] .= 10.0
#flow[:,:, 1] +=rand(H,W)*4
#flow[:,:, 2] +=rand(H,W)*4

#Wop = compute_warping_operator(flow)
#img1_ = Wop * vec(img2)
#img1 = reshape(img1_, H, W)

us = zeros(H,W,2)
us[:,:,2] = img2
us[:,:,1] = img1

v = get_flows(us)

W_list = mapslices(f -> compute_warping_operator(f), v,dims=[1,2,3])
@info size(W_list)
#t = 1
#Wuv .= mul!(Wuv, W_list[t], vec(us[:,:,t+1])) .- vec(us[:,:,t])

#flow_x, flow_y, im2Warped = py"py_flow"(img1,img2)
#v = zeros(H,W,2)
#v[:,:,1] = flow_x
#v[:,:,2] = flow_y#W_list = mapslices(f -> compute_warping_operator(f), v, dims=[1,2,3])
#Woptical = W_list[1]#compute_warping_operator(v)
#img3_ = Woptical * vec(img2)
#img3 = reshape(img3_,H, W)
#W_list = mapslices(f -> compute_warping_operator(f), v,dims=[1,2,3])



t = 1
_ubar_1 = view(us,:,:,t+1)
_ubar = view(us,:,:,t)
Wu = W_list[t]*(collect(Iterators.flatten(_ubar_1))) - (collect(Iterators.flatten(_ubar)))
p3_ascent = reshape(Wu, H, W)

p3_adjoint = reshape(W_list[t]'*vec(p3_ascent),H,W)-p3_ascent

img3 = max.(us[:,:,1].-p3_adjoint, 0.0)

p1 = plot(Gray.(img1[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="img1")
p2 = plot(Gray.(img2[:,:,1]), aspect_ratio=:equal, framestyle=:none, title="img2")
p3 = plot(Gray.(img3), aspect_ratio=:equal, framestyle=:none, title="rec")
p4 = plot(Gray.(p3_ascent), aspect_ratio=:equal, framestyle=:none, title="ascent")
p5 = plot(Gray.(p3_adjoint), aspect_ratio=:equal, framestyle=:none, title="adjoint")
l = @layout [a b c d e]
plot(p1, p2, p3, p4, p5, layout = l, size=(2000,600), linewidth=5)
#XX, YY = repeat(collect(1:W)', H, 1), repeat(collect(1:H), 1, W)
#p1 = quiver([(XX...)...][1:20:end], [(YY...)...][1:20:end], quiver=([(v[:,:,1]...)...][1:20:end]*0.001, [(v[:,:,2]...)...][1:20:end]*0.001), title="v")
#p2 = quiver([(XX...)...][1:20:end], [(YY...)...][1:20:end], quiver=([(flow[:,:,1]...)...][1:20:end]*0.1, [(flow[:,:,2]...)...][1:20:end]*0.1), title="flow")
#l = @layout [a b]
#plot(p1, p2, layout = l, size=(2000,600), linewidth=5)

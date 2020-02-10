#### USING PACKAGE PYFLOW ###################################

using PyCall
using ProgressMeter

py"""
import numpy as np
import pyflow

def py_flow(img1,img2,alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=1, nInnerFPIterations=1, nSORIterations=30, colType=1):
    img1 = np.array(img1)
    img2 = np.array(img2)
    height, width = img1.shape

    im1 = np.array(img1).reshape(height,width,1).copy(order='C')
    im2 = np.array(img2).reshape(height,width,1).copy(order='C')

    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)

    return u, v, im2W
"""
##############################################################

function get_flows(u::Array{T,3})::Array{T,4} where {T<:AbstractFloat}
    height, width, frames = size(u)
    result = zeros(height, width, 2, frames)
    for i in 1:frames-1
        flow_x, flow_y, im2Warped = py"py_flow"(u[:,:,i],u[:,:,i+1])
        result[:,:,1,i] = flow_x
        result[:,:,2,i] = flow_y
    end
    return result
end

#######################################################################################
using SparseArrays
using LinearOperators

### Original python impl - think something wrong here
# py"""
# import numpy as np
#
# def sub2ind(array_shape, rows, cols, is_vec=False):
#     out = rows*array_shape[1] + cols
#
#     if is_vec:
#         return out.reshape(-1)
#     else:
#         return out
#
# def py_warping_operator(flow):
#     flow = np.array(flow)
#     H, W, dim = flow.shape
#     shape = (H,W)
#     XX, YY = np.meshgrid(np.arange(W), np.arange(H))
#
#     XX_target = XX + flow[:,:,0]
#     YY_target = YY + flow[:,:,1]
#
#     x1 = (np.floor(XX_target) - 1).astype(np.int32)
#     x2 = x1 + 1
#     x3 = x1 + 2
#     x4 = x1 + 3
#
#     y1 = (np.floor(YY_target) - 1).astype(np.int32)
#     y2 = y1 + 1
#     y3 = y1 + 2
#     y4 = y1 + 3
#
#     v2 = YY_target - y2
#     v1 = XX_target - x2
#
#     indicator = np.ones([H,W], dtype=np.int32)
#     indicator[x1 < 0] = 0
#     indicator[x4 >= W] = 0
#     indicator[y1 < 0] = 0
#     indicator[y4 >= H] = 0
#
#     indicator = indicator > 0
#     v1 = v1[indicator]
#     v2 = v2[indicator]
#
#     idxList = sub2ind(shape, YY, XX)
#     idxList_ = np.reshape(idxList, -1)
#
#     # find the indices
#     list_J  = np.stack([
#         idxList_[sub2ind(shape, y2[indicator], x2[indicator], False)],
#         idxList_[sub2ind(shape, y2[indicator], x3[indicator], False)],
#         idxList_[sub2ind(shape, y3[indicator], x2[indicator], False)],
#         idxList_[sub2ind(shape, y3[indicator], x3[indicator], False)]] )
#
#     list_J = np.reshape(list_J, -1)
#
#     list_val  = [ ( (1.-v1) * (1.-v2)) ]
#     list_val += [ (v2 * (1.-v1))]
#     list_val += [ (v1 * (1.-v2))]
#     list_val += [ (v1 * v2)]
#
#     list_val_stack = np.stack(list_val)
#     list_val = np.reshape(list_val_stack, -1)
#
#     list_I = idxList[indicator]
#     list_I = np.repeat(np.reshape(list_I, [1,-1]), 4, axis=0).reshape(-1)
#
#     return list_I, list_J, list_val
# """

function sub2ind(array_shape, rows, cols)
    out = (cols.-1).*array_shape[1] .+ rows
    return convert.(Int64, out)
end

function compute_warping_operator(flow)
    # ref: https://github.com/HendrikMuenster/flexBox/blob/79abc7285703911cca2653434cca5bcefc79c722/operators/generateMatrix/warpingOperator.m
    """
    u: gray image
    v: flow [H, W, 2]
    """
    H, W, dims = size(flow)
    shape = (H,W)
    XX, YY = repeat(collect(1:W)', H, 1), repeat(collect(1:H), 1, W)

    targetPoint2Mat = XX + flow[:,:,1]
    targetPoint1Mat = YY + flow[:,:,2]

    x1 = floor.(targetPoint1Mat).-1
    x2 = x1 .+ 1
    x3 = x1 .+ 2
    x4 = x1 .+ 3

    y1 = floor.(targetPoint2Mat).-1
    y2 = y1 .+ 1
    y3 = y1 .+ 2
    y4 = y1 .+ 3

    v2 = targetPoint1Mat .- x2
    v1 = targetPoint2Mat .- y2

    indicator = ones(H,W)
    indicator = (x1 .> 0).*indicator
    indicator = (x4 .<= W).*indicator
    indicator = (y1 .> 0).*indicator
    indicator = (y4 .<= H).*indicator

    indicator = indicator .> 0
    v1 = v1[indicator]
    v2 = v2[indicator]

    idxList = sub2ind(shape, YY, XX)
    idxList_ = collect(Iterators.flatten(idxList))

    # find the indices
    # list_J  = vcat(
    #     idxList_[sub2ind(shape, x2[indicator], y2[indicator])],
    #     idxList_[sub2ind(shape, x3[indicator], y2[indicator])],
    #     idxList_[sub2ind(shape, x2[indicator], y3[indicator])],
    #     idxList_[sub2ind(shape, x3[indicator], y3[indicator])])

    list_J  = vcat(
        idxList_[sub2ind(shape, x1[indicator], y1[indicator])],
        idxList_[sub2ind(shape, x2[indicator], y1[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y1[indicator])],
        idxList_[sub2ind(shape, x4[indicator], y1[indicator])],
        idxList_[sub2ind(shape, x1[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x2[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x4[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x1[indicator], y3[indicator])],
        idxList_[sub2ind(shape, x2[indicator], y3[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y3[indicator])],
        idxList_[sub2ind(shape, x4[indicator], y3[indicator])],
        idxList_[sub2ind(shape, x1[indicator], y4[indicator])],
        idxList_[sub2ind(shape, x2[indicator], y4[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y4[indicator])],
        idxList_[sub2ind(shape, x4[indicator], y4[indicator])])

    list_J = collect(Iterators.flatten(list_J))

    # list_val  = ( (1 .-v1) .* (1 .-v2))
    # list_val = vcat(list_val,(v2 .* (1 .-v1)))
    # list_val = vcat(list_val,(v1 .* (1 .-v2)))
    # list_val = vcat(list_val,(v1 .* v2))
    list_val =vcat(
        (v1.*v2.*(v1 .- 1).^2 .*(v2 .- 1).^2)/4,
        -1 .*(v1.*(v1 .- 1).^2 .*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
        -1 .*(v1.*v2.*(v1 .- 1).^2 .*(- 3 .*v2.^2 + 4 .*v2 .+ 1))./4,
        -1 .*(v1.*v2.^2 .*(v1 .- 1).^2 .*(v2 .- 1))./4,
        -1 .*(v2.*(v2 .- 1).^2 .*(3 .*v1.^3 .- 5 .*v1.^2 .+ 2))./4,
        ((3 .*v1.^3 .- 5 .*v1.^2 .+ 2).*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
        (v2.*(- 3 .*v2.^2 .+ 4 .*v2 .+ 1).*(3 .*v1.^3 .- 5 .*v1.^2 .+ 2))./4,
        (v2.^2 .*(v2 .- 1).*(3 .*v1.^3 .- 5 .*v1.^2 .+ 2))./4,
        -1 .*(v1.*v2.*(v2 .- 1).^2 .*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1))./4,
        (v1.*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1).*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
        (v1.*v2.*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1).*(- 3 .*v2.^2 .+ 4 .*v2 .+ 1))./4,
        (v1.*v2.^2 .*(v2 .- 1).*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1))./4,
        -1 .*(v1.^2 .*v2.*(v1 .- 1).*(v2 .- 1).^2)./4,
        (v1.^2 .*(v1 .- 1).*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
        (v1.^2 .*v2.*(v1 .- 1).*(- 3 .*v2.^2 .+ 4 .*v2 .+ 1))./4,
        (v1.^2 .*v2.^2 .*(v1 .- 1).*(v2 .- 1))./4
    )

    list_val = collect(Iterators.flatten(list_val))

    list_I = idxList[indicator]
    list_I =collect(Iterators.flatten(repeat(list_I,1,16)))

    # list_I_py, list_J_py, list_val_py = py"py_warping_operator"(flow)
    # list_I_py = list_I_py.+1
    # list_J_py = list_J_py.+1
    # W_mat_py = sparse(list_I_py, list_J_py, list_val_py, H*W, H*W)

    W_mat = sparse(list_I, list_J, list_val, H*W, H*W)

    return LinearOperator(W_mat)
end

py"""
import numpy as np
def proj_dual_l1(x, radius=1.0):
    proj = np.sign(x) * np.minimum(np.absolute(x), radius)
    return proj
"""

function _recon2d_tv_primaldual_flow(As,u0s,bs,w_tv,w_flow,c,v,niter)
    height, width, frames = size(u0s)

    u = u0s
    ubar = deepcopy(u)
    #
    p1 = zeros(size(bs))
    p2 = zeros(height, width, 2,frames)
    p3 = zeros(height, width, frames)

    A_A_norm = map(A -> (A,compute_opnorm(A)), As)
    println(size(v))
    W_list = mapslices(f -> compute_warping_operator(f), v,dims=[1,2,3])
    #println(size(A_A_norm))
    p_adjoint  = zeros(height, width, frames)
    p3_adjoint = zeros(height, width, frames)
    for it=1:niter
        tau = 1.0
        u_prev = deepcopy(u)
        for t=1:frames
            A, A_norm = A_A_norm[t]
            opsnorm = [A_norm, sqrt(8), sqrt(8)]
            sigmas = map(n-> 1.0/(n*c), opsnorm)

            tau = c / sum(opsnorm)
            ops = [A,D]
            data1, data2, p1[:,t], p2[:,:,:,t], p1_adjoint, p2_adjoint = get_tv_adjoints!(ops, bs[:,t], ubar[:,:,t], w_tv, sigmas[1:2], p1[:,t], p2[:,:,:,t], height, width)

            p_adjoint[:,:,t] = p1_adjoint + p2_adjoint

            if t < frames
                # I2_warp - I1
                Wu = W_list[t]*(collect(Iterators.flatten(ubar[:,:,t+1]))) - (collect(Iterators.flatten(ubar[:,:,t])))

                p3_ascent = p3[:,:,t] + sigmas[3] * reshape(Wu, height, width)
                p3[:,:,t] = py"proj_dual_l1"(p3_ascent, w_flow)

                #energy_flow_data += np.sum(np.absolute(Wu))

                # \mathcal{W}^T = [-I; W^T]
                p3_adjoint_t1 = W_list[t]'*collect(Iterators.flatten(p3[:,:,t]))
                p3_adjoint[:,:,t] += reshape(p3_adjoint_t1, height,width)
                p3_adjoint[:,:,t] += -p3[:,:,t]
            end
        end

        # primal update
        u_descent = u - tau*p_adjoint
        u = max.(u_descent, 0.0) # positivity constraint

        # acceleration
        ubar = 2*u - u_prev

        # compute primal energy (optional)
        # if it % 10 == 0
        #     energy = sum(data1.^2) / length(data1) + sum(abs.(data2)) / length(data2)
        #     println("$it, approx. primal energy: $energy")
        # end
    end
    return u
end

"""
    recon2d_tv_primaldual_flow(As, bs, u0s, niter, w_tv, w_flow, c=1.0)

Reconstruct a 2d image by total variation model using Primal Dual optimization method

# Args
As : List of forward opeartors
bs : List of projection data
u0s: Initial guess of images
niter: number of iterations
w_tv: weight for TV term
W_flow: weight for flow term
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica

"""
function recon2d_tv_primaldual_flow(As, bs::Array{R, 2}, u0s::Array{R, 3}, niter::Int, w_tv::R, w_flow::R, c=1.0) where {R <: AbstractFloat}
    height, width, frames = size(u0s)
    v = zeros(height, width, 2, frames)
    u = u0s
    p = Progress(niter,1)
    for i = 1:niter
        u_prev = u
        v_prev = v
        u = _recon2d_tv_primaldual_flow(As,u,bs,w_tv,w_flow,c,v,10)
        v = get_flows(u)
        next!(p)
    end
    return u
end

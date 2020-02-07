#### USING PACKAGE PYFLOW ###################################

using PyCall

py"""
import numpy as np
import pyflow

def flow(img1,img2,alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=1, nInnerFPIterations=1, nSORIterations=30, colType=1):
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
        flow_x, flow_y, im2Warped = py"flow"(u[:,:,i],u[:,:,i+1])
        result[:,:,1,i] = flow_x
        result[:,:,2,i] = flow_y
    end
    return result
end

#######################################################################################
function _recon2d_tv_primaldual_flow(As,u0s,bs,w_tv,w_flow,c,v,niter)
    height, width, frames = size(u0s)

    u = u0s
    ubar = deepcopy(u)
    #
    p1 = zeros(size(bs))
    p2 = zeros(height, width, 2,frames)

    A_A_norm = map(A -> (A,compute_opnorm(A)), As)
    #println(size(A_A_norm))
    p_adjoint  = zeros(height, width, frames)
    for it=1:niter
        tau = 1.0
        u_prev = deepcopy(u)
        for t=1:frames
            A, A_norm = A_A_norm[t]
            opsnorm = [A_norm, sqrt(8), sqrt(8)]
            sigmas = map(n-> 1.0/(n*c), opsnorm)

            tau = c / sum(opsnorm)
            ops = [A,D]
            data1, data2, p1[:,:,t], p2[:,:,:,t], p1_adjoint, p2_adjoint = get_tv_adjoints!(ops, bs[:,t], ubar[:,:,t], w_tv, sigmas[1:2], p1[:,:,t], p2[:,:,:,t], height, width)

            p_adjoint[:,:,t] = p1_adjoint + p2_adjoint

            if t < frames
                data3 = ubar[:,:,t+1]-ubar[:,:,t]+coordinate_dot(ops[2]*ubar[:,:,t],v[:,:,:,t])
                p3[:,:,t] += sigmas[3]*data3
                proj_dual_iso!(p3[:,:,t], w_flow)
                p3_adjoint = (p3[:,:,t+1])'-(p3[:,:,t])'+coordinate_dot(permutedims(v[:,:,t], [3,2,1]),G'*p3[:,:,t])
                p_adjoint[:,:,t] +=p3_adjoint
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
    for i = 1:niter
        u_prev = u
        v_prev = v
        u = _recon2d_tv_primaldual_flow(As,u0s,bs,w_tv,w_flow,c,v,800)
        v = get_flows(u)
    end
    return u
end

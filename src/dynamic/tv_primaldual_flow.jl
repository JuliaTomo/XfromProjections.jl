using .optical_flow
using .util_convexopt
using ProgressMeter
using Suppressor

function _recon2d_tv_primaldual_flow(As,A_norm,W_list,W_norm,u0s,bs,w_tv,w_flow,c,niter)
    #A variational reconstruction method for undersampled dynamic x-ray tomography based on physical motion models (Burger)#
    height, width, frames = size(u0s)

    u = u0s
    ubar = deepcopy(u)
    #
    p1 = zeros(size(bs))
    p2 = zeros(height, width, 2,frames)
    p3 = zeros(height, width, frames)


    p_adjoint  = zeros(height, width, frames)
    p3_adjoint = zeros(height, width, frames)
    opsnorm = [A_norm, sqrt(8), W_norm]
    tau = c /sum(opsnorm)
    for it=1:niter
        u_prev = deepcopy(u)
        energy_data = 0.
        energy_tv = 0.
        energy_flow_data = 0.
        for t=1:frames
            A = As[t]
            sigmas = map(n-> 1.0/(n*c), opsnorm)

            ops = [A,D]
            data1, data2, p1[:,t], p2[:,:,:,t], p1_adjoint, p2_adjoint = get_tv_adjoints!(ops, bs[:,t], ubar[:,:,t], w_tv, sigmas[1:2], p1[:,t], p2[:,:,:,t], height, width)
            p_adjoint[:,:,t] = p1_adjoint + p2_adjoint

            if t < frames
                Wu = W_list[t]*(collect(Iterators.flatten(ubar[:,:,t+1]))) - (collect(Iterators.flatten(ubar[:,:,t])))
                p3_ascent = p3[:,:,t] + sigmas[3] * reshape(Wu, height, width)
                p3[:,:,t] = proj_dual_l1(p3_ascent, w_flow)

                p3_adjoint_t1 = W_list[t]'*collect(Iterators.flatten(p3[:,:,t]))
                p3_adjoint[:,:,t] += reshape(p3_adjoint_t1, height,width)
                p3_adjoint[:,:,t] += -p3[:,:,t]
            end
            p_adjoint[:,:,t] += p3_adjoint[:,:,t]
        end

        # primal update
        u_descent = u - tau*p_adjoint
        u = max.(u_descent, 0.0) # positivity constraint

        # acceleration
        ubar = 2*u - u_prev

    end
    return u
end

"""
    recon2d_tv_primaldual_flow(As, bs, u0s, niter, w_tv, w_flow, c=1.0)

Reconstruct a 2d image by total variation model using Primal Dual optimization method and optical flow estimation

# Args
As : List of forward opeartors
bs : List of projection data
u0s: Initial guess of images
niter: number of iterations
w_tv: weight for TV term
W_flow: weight for flow term
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica

"""
function recon2d_tv_primaldual_flow(A_list, bs::Array{R, 2}, u0s::Array{R, 3}, niter1::Int, niter2::Int, w_tv::R, w_flow::R, c=1.0) where {R <: AbstractFloat}
    height, width, frames = size(u0s)
    v = zeros(height, width, 2, frames)
    u = u0s
    shape = (height*width,frames)
    A_norm = compute_opnorm_block(A_list, shape)
    p = Progress(niter1,1, "Optimizing")
    for i = 1:niter1
        u_prev = u
        v_prev = v
        W_list = mapslices(f -> compute_warping_operator(f), v,dims=[1,2,3])
        @suppress begin
            W_norm = compute_opnorm_block(W_list, shape)
            u = _recon2d_tv_primaldual_flow(A_list,A_norm,W_list,W_norm,u,bs,w_tv,w_flow,c,niter2)
        end
        v = get_flows(u)
        next!(p)
    end
    return u
end

using .util_convexopt

@doc raw"""
Solve u

ops[1]: A
ops[2]: TV

argmin_u \|ops[1]*u - b\|_2^2 + w_tv TV(u)
"""
function _recon2d_tv_primaldual(ops, b, u0, niter, w_tv, sigmas, tau)
    H, W = size(u0)
    nops = length(ops)
    
    u = u0
    ubar = deepcopy(u)

    p1 = zeros(size(b))
    p2 = zeros(H, W, 2)

    for it=1:niter
        u_prev = deepcopy(u)

        # dual update
        data1 = (ops[1]*vec(ubar) .- b)
        # p1 = proj_dual_l1(p1_ascent, w_data) # l1 norm
        p1 = (p1 + sigmas[1] * data1) ./ (sigmas[1] + 1.0) # l2 norm
        p1_adjoint = reshape(ops[1]' * p1, H, W)

        data2 = ops[2]*ubar
        p2 = p2 + sigmas[2]*data2
        proj_dual_iso!(p2, w_tv)
        p2_adjoint = ops[2]' * p2

        
        p_adjoint = p1_adjoint + p2_adjoint

        # primal update
        u_descent = u - tau*p_adjoint
        u = max.(u_descent, 0.0) # positivity constraint

        # acceleration
        ubar = 2*u - u_prev

        # compute primal energy (optional)
        if it % 10 == 0
            energy = sum(data1.^2) / length(data1) + sum(abs.(data2)) / length(data2)
            println("$it, approx. primal energy: $energy")
        end
    end
    return u
end

"""
    recon2d_tv_primaldual(A, b, u0, niter, w_tv, c=1.0)

Reconstruct a 2d image by total variation model using Primal Dual optimization method

# Args
A : Forward opeartor
b : Projection data 
u0: Initial guess of image
niter: number of iterations
w_tv: weight for TV term
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica

"""
function recon2d_tv_primaldual(A, b::Array{R, 1},
    u0::Array{R, 2}, niter::Int, w_tv::R, c=1.0) where {R <: AbstractFloat}

    ops = [A, D]
    @time op_A_norm = compute_opnorm(A)
    println("@ opnorm of forward projection operator: $op_A_norm")
    ops_norm = [op_A_norm, sqrt(8)]
    
    sigmas = zeros(length(ops))
    for i=1:length(ops)
        sigmas[i] = 1.0 / (ops_norm[i] * c)
    end

    tau = c / sum(ops_norm)

    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")
    
    return _recon2d_tv_primaldual(ops, b, u0, niter, w_tv, sigmas, tau)
end
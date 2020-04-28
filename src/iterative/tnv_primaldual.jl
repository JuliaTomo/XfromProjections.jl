using .util_convexopt

"""
    function prox_nuclear!(y, x, stepsize)

Proximal operator of nuclear norm to x. y=prox_{s f}(x)
"""
function prox_nuclear_wo_svd!(y, x, stepsize)
    H, W, M, C = size(x) # Dim : 2 for 2D, 3 for 3D (C >= 2)
    x_HxWxCx2 = PermutedDimsArray(x, [1,2,4,3])
    y_HxWxCx2 = PermutedDimsArray(y, [1,2,4,3])
    x_ = reshape(x_HxWxCx2, H*W, C, 2)
    y_ = reshape(y_HxWxCx2, H*W, C, 2)

    EPS = 1e-8

    # x_svd = mapslices(svd!, x_, dims=[2,3])
    # x_svd = dropdims(x_svd_, dims=4)
    Threads.@threads for hw=1:H*W
        x_hw = view(x_, hw, :, :) # [C x 2]
        y_hw = view(y_, hw, :, :)

        BB1 = 0.0
        BB2 = 0.0
        BB4 = 0.0
        
        for k=1:C
            BB1 += x_hw[k, 1] * x_hw[k, 1]
            BB2 += x_hw[k, 1] * x_hw[k, 2]
            BB4 += x_hw[k, 2] * x_hw[k, 2]
        end

        # # eigenvalues of BB
        T = BB1 + BB4
        D = BB1 * BB4 - BB2 * BB2

        det = sqrt( max( ( T*T / 4.0) - D, 0.0) )
        
        eig1 = max(  (T / 2.0) + det , 0.0 )
        eig2 = max(  (T / 2.0) - det , 0.0 )
        sigma1 = sqrt(eig1)
        sigma2 = sqrt(eig2)

        if abs(BB2) > EPS
            v0 = BB2
            V1 = eig1 - BB4
            V2 = eig2 - BB4

            len1 = sqrt(v0*v0 + V1*V1)
            len2 = sqrt(v0*v0 + V2*V2)

            if len1 > EPS
                V1 /= len1
                V3 = v0 / len1
            end
            if len2 > EPS
                V2 /= len2
                V4 = v0 / len2
            end
        else
            if BB1 > BB4
                V1 = V4 = 1.0
                V2 = V3 = 0.0
            else
                V1 = V4 = 0.0
                V2 = V3 = 1.0
            end
        end

        ss1 = max(sigma1 - stepsize, 0.0)
        ss2 = max(sigma2 - stepsize, 0.0)
        if sigma1 > EPS
            ss1 /= sigma1
        end
        if sigma2 > EPS
            ss2 /= sigma2
        end

        t1 = ss1 * V1 * V1 + ss2 * V2 * V2
        t2 = ss1 * V1 * V3 + ss2 * V2 * V4
        t3 = ss1 * V3 * V3 + ss2 * V4 * V4
        
        # B * t
        for c=1:C
            y_hw[c,1] = x_hw[c,1]*t1 + x_hw[c,2]*t2
            y_hw[c,2] = x_hw[c,1]*t2 + x_hw[c,2]*t3
        end
    end
    return y
end

function grad2d!(du, u) where {T<:AbstractFloat}
    du1, du2 = view(du, :, :, 1), view(du, :, :, 2)
    du1 .= circshift!(du1, u, [0, -1]) .- u
    du2 .= circshift!(du2, u, [-1, 0]) .- u
    du1[:, end] .= 0.0
    du2[end, :] .= 0.0
end

function div2d!(divp, p) where {T<:AbstractFloat}
    p1, p2 = view(p,:,:,1), view(p,:,:,2)
    p1_x, p2_y = view(divp,:,:,1), view(divp,:,:,2)
    
    # p2_y is temp variable to save memory
    p1_x         .=  p1 - circshift!(p2_y, p1, [0, 1])
    p1_x[:, end] .= -p1[:, end-1]
    p1_x[:,   1] .=  p1[:, 1]
    
    circshift!(p2_y, p2, [1, 0])
    p2_y         .= p2 - circshift!(p2_y, p2, [1, 0])
    p2_y[end, :] .= -p2[end-1, :]
    p2_y[1,   :] .=  p2[1, :]
    
    p1_x .+= p2_y
    return p1_x
end

function _recon2d_tnv_primaldual!(u::Array{T, 3}, A, b0::Array{T, 3}, niter, w_data, sigmas, tau) where {T<:AbstractFloat}
    At = A'
    H, W, C = size(u)
    
    b = reshape(b0, (size(b0, 1)*size(b0, 2), C))
    # b = vec(b0)
    
    ubar = deepcopy(u)
    u_prev = similar(u)

    p1 = zeros(size(b))
    p2 = zeros(H, W, 2, C) # q in Duran 2016
    g  = similar(p2)
    p2_temp = similar(p2)

    p_adjoint = zeros(H, W, C)

    du = similar(p2)
    divp2 = similar(p2)
    
    data1 = similar(b)

    invsigma11 = 1. / (sigmas[1] / w_data + 1.0) # l2 data, (Handa Sec 8.1)
    invsigma2 = 1. / sigmas[2]

    for it=1:niter
        u_prev .= u

        Threads.@threads for c=1:C
            ubar_c = view(ubar, :, :, c)
            data1_c = view(data1, :, c)
            p_adjoint_c = vec(view(p_adjoint, :, :, c))
            p1_c = view(p1, :, c)

            # dual update: data fidelity
            data1_c .= mul!(data1_c, A, vec(ubar_c)) .- view(b, :, c)
            p1_c .= (p1_c .+ sigmas[1] .* data1_c) .* invsigma11
            mul!(p_adjoint_c, At, p1_c)
            
            # compute gradient for TNV later
            du_c = view(du, :, :, :, c)     
            grad2d!(du_c, ubar_c)
        end

        # projection onto dual of (S1,l1)
        p2_temp .= du .+ invsigma2 .* p2
        prox_nuclear_wo_svd!(g, p2_temp, invsigma2)
        # prox_nuclear!(g, p2_temp, invsigma2)
        p2 .+= sigmas[2] .* ( du .- g )

        Threads.@threads for c=1:C
            p_adjoint_c = view(p_adjoint, :, :, c)
            p_adjoint_c .-= div2d!(view(p2_temp,:,:,:,c), view(p2,:,:,:,c))
        end
            
        # primal update
        u .= max.(u .- tau .* p_adjoint, 0.0) # positivity constraint

        # acceleration
        ubar .= 2 .* u .- u_prev

        # compute primal energy (optional)
        if it % 20 == 0
            energy = sum(data1.^2) / length(data1)
            println("$it, approx. data term: $energy")
        end
    end
    return u
end

"""
    recon2d_tv_primaldual!(u::Array{T, 2}, A, b::Array{T, 2}, niter::Int, w_tv::T, c=1.0)

Reconstruct a 2d image by TV-L2 model using Primal Dual optimization method

# Args
u : Initial guess of images
A : Forward opeartor
b : Projection data 
niter: number of iterations
w_tv: weight for TV term
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica
"""
function recon2d_tnv_primaldual!(u::Array{T, 3}, A, b::Array{T, 3}, niter::Int, w_data::T, c=1.0) where {T <: AbstractFloat}
    if size(u, 3) != size(b, 3)
        error("The channel size of u and b should match.")
    end

    op_A_norm = compute_opnorm(A, 6) # for safety
    println("@ opnorm of forward projection operator: $op_A_norm")
    ops_norm = [op_A_norm, sqrt(8)]
    
    sigmas = zeros(length(ops_norm))
    for i=1:length(ops_norm)
        sigmas[i] = 1.0 / (ops_norm[i] * c)
    end

    tau = c / sum(ops_norm)
    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")
    
    return _recon2d_tnv_primaldual!(u, A, b, niter, w_data, sigmas, tau)
end


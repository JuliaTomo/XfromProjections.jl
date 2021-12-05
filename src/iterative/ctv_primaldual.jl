using .util_convexopt

"project a vector 'u' to l1 unit ball "
function proj_l1_ball!(u::Array{T, 1}) where {T<:AbstractFloat}
    sum = Inf
    shrink = 0.0
    while (sum > 1.0)
        sum = 0.0
        cnt = 0

        for i=1:length(u)
            u[i] = max(u[i] - shrink, 0.0)
            sum += abs(u[i])
            if u[i] != 0.0
                cnt += 1
            end
        end
        if cnt > 0
            shrink = (sum - 1.0) / cnt
        else
            break
        end
    end
end
 
"""
    function prox_nuclear!(y, x, stepsize)

Proximal operator of nuclear norm to x. y=prox_{s f}(x)

is_S1=true : (S1, l1) norm for TNV
is_S1=false: (S∞, l1)
"""
function prox_nuclear_wo_svd!(y, x, stepsize, is_S1=true)
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
        σ1 = sqrt(eig1)
        σ2 = sqrt(eig2)

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

        if is_S1
            # (S1, l1) Eq. (11) in IPOL paper
            ss1 = max(σ1 - stepsize, 0.0)
            ss2 = max(σ2 - stepsize, 0.0)
        else
            # (S∞, l1) prox to l∞
            # l∞(x) = x - λ prox_(*/λ) (x / λ)
            # proj_dual_l1!(σ1/sigma, 1.0/sigma)
            # proj_dual_l1: x ./= max.(1.0, abs.(x) / weight)
            pp = [σ1 / stepsize, σ2 / stepsize]
            proj_l1_ball!(pp)
            
            ss1 = σ1 - stepsize * pp[1]
            ss2 = σ2 - stepsize * pp[2]
        end

        if σ1 > EPS
            ss1 /= σ1
        end
        if σ2 > EPS
            ss2 /= σ2
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

"""
Example:
x = rand(20, 30, 2, 10);
y = zeros(20, 30, 2, 10);
prox_l∞11!(y, x, 0.01);
"""
function prox_l∞11!(y, x, stepsize)
    H, W, M, C = size(x) # Dim : 2 for 2D, 3 for 3D (C >= 2)
    x_HxWxCx2 = PermutedDimsArray(x, [1,2,4,3])
    y_HxWxCx2 = PermutedDimsArray(y, [1,2,4,3])
    x_ = reshape(x_HxWxCx2, H*W, C, 2)
    y_ = reshape(y_HxWxCx2, H*W, C, 2)

    # x_svd = mapslices(svd!, x_, dims=[2,3])
    # x_svd = dropdims(x_svd_, dims=4)
    Threads.@threads for hw=1:H*W
        x_hw = view(x_, hw, :, :) # [C x 2]
        y_hw = view(y_, hw, :, :)

        pp1 = zeros(C)
        pp2 = zeros(C)
        for c=1:C
            pp1[c] = abs(x_hw[c,1]) / stepsize
            pp2[c] = abs(x_hw[c,2]) / stepsize
        end

        proj_l1_ball!(pp1)
        proj_l1_ball!(pp2)

        for c=1:C
            x1 = x_hw[c,1]
            x2 = x_hw[c,2]

            y_hw[c,1] = x1 - stepsize * sign(x1) * pp1[c]
            y_hw[c,2] = x2 - stepsize * sign(x2) * pp2[c]
        end
    end
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

function _recon2d_ctv_primaldual!(u::Array{T, 3}, A, b0::Array{T, 3}, niter, w_data, sigmas, tau, type, ϵ, nverbose) where {T<:AbstractFloat}
    At = A'
    H, W, C = size(u)

    # normalize the data fidelity with respect.
    # we consider mean(residual) instead of sum
    w_data = w_data / C
    #w_data = w_data / (C  )
    
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
    
    data1 = copy(b)

    invσ11 = 1. / (sigmas[1] / w_data + 1.0) # l2 data, (Handa Sec 8.1)
    invσ2 = 1. / sigmas[2]

    res_primals = zeros(niter)
    res_duals = zeros(niter)

    if nverbose > 0
        p_adjoint_prev = similar(p_adjoint)
    #    if flag_dual_residual == true
            # variables for computinig residuals
    
            p1_prev = similar(p1)
            p2_prev = similar(p2)
            du_prev = similar(du)
            data1_prev = similar(data1)
        # end
    end

    # init p1
    # Threads.@threads for c=1:C
    #     @views mul!(data1[:,c], A, vec(ubar[:,:,c]))
    #     p1[:,c] .= data1[:,c]
    #     @views data1[:,c] .-= b[:,c]
    # end

    for it=1:niter
        if nverbose > 0 && it % nverbose == 0
            copy!(p_adjoint_prev, p_adjoint)
            # if flag_dual_residual == true
                copy!(data1_prev, data1)
                copy!(p1_prev, p1)
                copy!(p2_prev, p2)
                copy!(du_prev, du)
            # end
        end

        Threads.@threads for c=1:C
            ubar_c = view(ubar, :, :, c)
            data1_c = view(data1, :, c)
            p_adjoint_c = vec(view(p_adjoint, :, :, c))
            p1_c = view(p1, :, c)

            # dual update: data fidelity
            data1_c .= mul!(data1_c, A, vec(ubar_c)) .- view(b, :, c)
            p1_c .= (p1_c .+ sigmas[1] .* data1_c) .* invσ11
            mul!(p_adjoint_c, At, p1_c)
            
            # compute gradient for TNV later
            du_c = view(du, :, :, :, c)     
            grad2d!(du_c, ubar_c)
        end

        u_prev .= u

        # -------- update dual p2
        # projection onto dual of tnv, or ..
        # prox_{lambda f*} (x) = x - lambda prox_{f / lambda} (x / lambda)
        # where x = r + sigma Du in the paper
        p2_temp .= du .+ invσ2 .* p2

        if type == "tnv" || type == "S1l1"
            prox_nuclear_wo_svd!(g, p2_temp, invσ2)
        elseif type == "S∞l1"
            prox_nuclear_wo_svd!(g, p2_temp, invσ2, false)
        elseif type == "l∞11" # strong coupling
            prox_l∞11!(g, p2_temp, invσ2)
        else
            error("not supported type")
        end

        # since x = r + sigma Du
        p2 .+= sigmas[2] .* ( du .- g )

        # -------- update primal
        Threads.@threads for c=1:C
            p_adjoint_c = view(p_adjoint, :, :, c)
            p_adjoint_c .-= div2d!(view(p2_temp,:,:,:,c), view(p2,:,:,:,c))
        end
            
        # primal update
        # u .= u .- tau .* p_adjoint
        u .= max.(u .- tau .* p_adjoint, 0.0) # positivity constraint

        # acceleration
        ubar .= 2 .* u .- u_prev

        # primal residual
        # if it > 1
        # end

        if nverbose > 0 && it % nverbose == 0
            # compute primal energy (optional)
            res_primal = sum(abs.((u_prev - u) / tau .- (p_adjoint_prev .- p_adjoint))) / length(u)
            res_primals[it] = res_primal

            residual = res_primal
            res_dual = 0.0
            
            # if flag_dual_residual == true
                res_dual1 = sum(abs.((p1_prev-p1) / sigmas[1] .- (data1_prev .- data1)))
                res_dual2 = sum(abs.((p2_prev-p2) / sigmas[2] .- (du_prev .- du)))
                res_dual = (res_dual1 + res_dual2) / (length(p1) + length(p2))    
                res_duals[it] = res_dual
            # end
            residual += res_dual
                        
            if residual < ϵ && it > 1
                @info "$it Stopping condition is met. $res_primal $res_dual"
                return (u, it, res_primals, res_duals)
            end

            energy = sum(data1.^2) /  length(data1)
            println("$it, data term: $energy, primal_res: $res_primal, dual_res: $res_dual")
        end
    end

    return (u, niter, res_primals, res_duals)
end

"""
    recon2d_ctv_primaldual!(u::Array{T, 2}, A, b::Array{T, 2}, niter::Int, w_tv::T, type="tnv", c=1.0)

Reconstruct a 2d image by Collaborative TV using Primal Dual optimization optimizer

# Args
u : Initial guess of images
A : Forward opeartor
b : Projection data 
niter : number of iterations
w_tv : weight for TV term
type (string) : tnv, S∞l1, l∞11
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica

For Collaborative TV, refer to:
Duran,Moeller,Sbert,Cremers_On_the_Implementation_of_Collaborative_TV_Regularization_-_Application_toImage_Processing_On_Line
"""
function recon2d_ctv_primaldual!(u::Array{T, 3}, A, b::Array{T, 3}, niter::Int, w_data, type="tnv", ϵ=1e-6, nverbose=10, c=1.0) where {T <: AbstractFloat}
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

    sigmas .*= 0.99
    tau *= 0.99

    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")
    
    return _recon2d_ctv_primaldual!(u, A, b, niter, w_data, sigmas, tau, type, ϵ, nverbose)
end


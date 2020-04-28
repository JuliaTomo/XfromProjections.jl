using .util_convexopt

function _recon2d_tv_primaldual!(u, A, b0, niter, w_tv, sigmas, tau)
    At = sparse(A')
    H, W = size(u)
    
    b = vec(b0)
    
    ubar = deepcopy(u)
    u_prev = similar(u)

    p1 = zeros(size(b))
    p2 = zeros(H, W, 2)

    p_adjoint = zeros(H, W)

    du = similar(p2)
    divp2 = similar(p2)
    
    data1 = similar(b)
    u_temp = similar(u)

    for it=1:niter
        u_prev .= u

        # dual update
        data1 .= mul!(data1, A, vec(ubar)) .- b
        # p1 = proj_dual_l1(p1_ascent, w_data) # l1 norm
        p1 .= (p1 .+ sigmas[1] * data1) ./ (sigmas[1] + 1.0) # l2 norm
        mul!(view(p_adjoint, :), At, p1)
        # p_adjoint .= reshape(At * p1, H, W)

        grad!(du, ubar)
        p2 .+= sigmas[2] .* du
        util_convexopt.proj_dual_iso!(p2, u_temp, w_tv)
        # util_convexopt.proj_dual_l1!(p2, w_tv) #anisotropic TV
        
        p_adjoint .-= div!(divp2, p2) # p1_adjoint + p2_adjoint
        
        # primal update
        u .= max.(u .- tau .* p_adjoint, 0.0) # positivity constraint

        # acceleration
        ubar .= 2 .* u .- u_prev

        # compute primal energy (optional)
        if it % 20 == 0
            energy = sum(data1.^2) / length(data1) + sum(abs.(du)) / length(du)
            println("$it, approx. primal energy: $energy")
        end
    end
    return u
end

"""
    recon2d_tv_primaldual!(u::Array{T, 2}, A, b, niter::Int, w_tv::T, c=1.0)

Reconstruct a 2d image by TV-L2 model using Primal Dual optimization method

# Args
u : Initial guess of image
A : Forward opeartor
b : Projection data (2 dimension or 1 dimension)
niter: number of iterations
w_tv: weight for TV term
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica
"""
function recon2d_tv_primaldual!(u::Array{T, 2}, A, b, niter::Int, w_tv::T, c=1.0) where {T <: AbstractFloat}
    @time op_A_norm = util_convexopt.compute_opnorm(A)
    println("@ opnorm of forward projection operator: $op_A_norm")
    ops_norm = [op_A_norm, sqrt(8)]
    
    sigmas = zeros(length(ops_norm))
    for i=1:length(ops_norm)
        sigmas[i] = 1.0 / (ops_norm[i] * c)
    end

    tau = c / sum(ops_norm)
    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")
    
    return _recon2d_tv_primaldual!(u, A, b, niter, w_tv, sigmas, tau)
end


function _recon2d_slices_tv_primaldual!(u::Array{T, 3}, A, b0::Array{T, 3}, niter::Int, w_tv::T, sigmas, tau::T) where {T<:AbstractFloat}
    H, W, nslice = size(u)
    
    At = sparse(A') # this significatnly improves the performance
    
    ubar = deepcopy(u)
    u_prev = similar(u)
    
    b_axWxH = permutedims(b0, [1, 3, 2]) # PermutedDimsArray(A, (3,1,2));
    b = reshape(b_axWxH, :, nslice)
    
    halfslice = Int(floor(nslice/2))

    p1 = zeros(size(b))
    p2 = zeros(H, W, nslice, 3)

    p_adjoint = zeros(H, W, nslice)
    du = similar(p2)
    divu = similar(p2)
    temp = zeros(size(A, 1), nslice)

    u_temp = similar(u)

    for it=1:niter
        u_prev .= u
        
        # dual update
        Threads.@threads  for slice=1:nslice
            ubar_slice = view(ubar, :, :, slice)
            p_adjoint_slice = vec(view(p_adjoint, :, :, slice))
            
            # for l2 norm
            @views temp[:, slice] .= mul!(temp[:, slice], A, vec(ubar_slice)) .- b[:, slice]
            @views p1[:, slice] .= (p1[:, slice] .+ sigmas[1] .* temp) ./ (sigmas[1] + 1.0)

            @views mul!(p_adjoint_slice, At, p1[:, slice])
        end

        # p2: 3d gradient and divergence
        grad!(du, ubar)
        p2 .+= sigmas[2] .* du
        temp_u = view(du)
        util_convexopt.proj_dual_iso!(p2, u_temp, w_tv)
        # util_convexopt.proj_dual_l1!(p2, w_tv) # anisotropic TV

        # if p_adjoint .++ -view(), memory allocation increases. WHY??
        p_adjoint .-= div!(divu, p2)

        # primal update
        u .= max.( u .- tau .* p_adjoint, 0.0) # positivity constraint

        # acceleration
        ubar .= 2.0 .* u .- u_prev

        # compute primal energy (optional)
        if it % 10 == 0
            # energy = sum(data1.^2) / length(data1) + sum(abs.(data2)) / length(data2)
            println("iter: $it, max: $(maximum(u)), $(maximum(p1))")
        end
    end
    return u
end

"""
    recon2d_tv_primaldual!(u::Array{T, 2}, A, b::Array{T, 2}, niter::Int, w_tv::T, c=1.0)

Reconstruct a 2d image by TV-L2 model using Primal Dual optimization method

# Args
- u : Initial guess of 3d image
- A : Forward opeartor
- b : Projection data [nangles x detheight x detwidth]
- niter: number of iterations
- w_tv: weight for TV term
- c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica
"""
function recon2d_slices_tv_primaldual!(u::Array{T, 3}, A, b::Array{T, 3}, niter::Int, w_tv::T, c=1.0) where {T <: AbstractFloat}
    @time op_A_norm = util_convexopt.compute_opnorm(A)
    println("@ opnorm of forward projection operator: $op_A_norm")
    ops_norm = [op_A_norm, sqrt(8)]
    println("TODO! in 3d, sqrt(8) would be wrong")
    
    sigmas = zeros(length(ops_norm))
    for i=1:length(ops_norm)
        sigmas[i] = 1.0 / (ops_norm[i] * c)
    end

    tau = c / sum(ops_norm)
    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")
    
    _recon2d_slices_tv_primaldual!(u, A, b, niter, w_tv, sigmas, tau)
end

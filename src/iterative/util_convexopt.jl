module util_convexopt

using LinearAlgebra
export LinOp, compute_opnorm, proj_l1!, grad!, div!, prox_nuclear!

# See http://www.ima.umn.edu/materials/2019-2020/SW10.14-18.19/28302/talk.pdf
# struct LinOp
#     forward::Function
#     adjoint::Function
# end
# Base.:(*)(A::LinOp, x) = A.forward(x) # for A*x
# Base.adjoint(A::LinOp) = LinOp(A.adjoint, A.forward) # for A'

# D_  = f -> grad(f)
# Dt_ = u -> -div(u)

# "D: gradient operator"
# D = LinOp(D_, Dt_)

# export D

"Compute opeartor norm based on the Power method"
function compute_opnorm(A::AbstractArray{T, 2}, niter=3) where {T<:Complex}
    lamb = 0
    x = rand(size(A,2))
    x_next = similar(x)
    Ax = zeros(size(A, 1))
    At = A'

    for i=1:niter
        x_next = A'*A*x # is very slow (WHY?) Todo
        #mul!(x_next, At, Ax)
        lamb_prev = lamb
        lamb = sqrt(sum(x_next.^2))

        if (abs(lamb_prev-lamb) < 1e-9)
            println("break at iteration: ", i)
            break
        end
        x = x_next / lamb
    end
    #eig_max = x'*A'*A*x / x*x
    return sqrt(lamb)
end

function compute_opnorm(A, niter=3)
    lamb = 0
    x = rand(size(A,2))
    x_next = similar(x)
    Ax = zeros(size(A, 1))
    At = A'

    for i=1:niter
        mul!(Ax, A, x)
        mul!(x_next, At, Ax)
        lamb_prev = lamb
        lamb = sqrt(sum(x_next.^2))

        if (abs(lamb_prev-lamb) < 1e-9)
            println("break at iteration: ", i)
            break
        end
        x = x_next / lamb
    end
    #eig_max = x'*A'*A*x / x*x
    return sqrt(lamb)
end

"Compute forward gradient [HxWx2] of an image `u` with Neumann boundary condition."
function grad(u::Array{T, 2}) where {T<:AbstractFloat}
    ux = circshift(u, [0, -1]) - u
    uy = circshift(u, [-1, 0]) - u
    ux[:, end] .= 0.0
    uy[end, :] .= 0.0

    return cat(ux, uy, dims=3)
end

function grad!(du::AbstractArray{T, 3}, u::AbstractArray{T, 2}) where {T<:AbstractFloat}
    du1, du2 = view(du, :, :, 1), view(du, :, :, 2)
    du1 .= circshift!(du1, u, [0, -1]) .- u
    du2 .= circshift!(du2, u, [-1, 0]) .- u
    du1[:, end] .= 0.0
    du2[end, :] .= 0.0
end

function grad!(du::Array{T, 4}, u::Array{T, 3}) where {T<:AbstractFloat}
    du1, du2, du3 = view(du,:,:,:,1), view(du,:,:,:,2), view(du,:,:,:,3)
    circshift!(du1, u, [0, -1, 0])
    du1 .= circshift!(du1, u, [0, -1, 0]) .- u
    du2 .= circshift!(du2, u, [-1, 0, 0]) .- u
    du3 .= circshift!(du3, u, [0, 0, -1]) .- u
    du1[:, end, :] .= 0.0
    du2[end, :, :] .= 0.0
    du3[:, :, end] .= 0.0
end


function div(p::Array{T, 3}) where {T<:AbstractFloat}
    return div2d(view(p,:,:,1), view(p,:,:,2))
end

function div!(divp::Array{T, 4}, p::Array{T, 4}) where {T<:AbstractFloat}
    p1, p2, p3 = view(p,:,:,:,1), view(p,:,:,:,2), view(p,:,:,:,3)
    p1_x, p2_y, p3_z = view(divp,:,:,:,1), view(divp,:,:,:,2), view(divp,:,:,:,3)

    # p3_z temp variable to save memory
    p1_x .= p1 .- circshift!(p3_z, p1, [0, 1, 0])
    p1_x[:, end, :] .= -p1[:, end-1, :]
    p1_x[:,   1, :] .=  p1[:, 1, :]

    p2_y .= p2 .- circshift!(p3_z, p2, [1, 0, 0])
    p2_y[end, :, :] .= -p2[end-1, :, :]
    p2_y[1,   :, :] .=  p2[1, :, :]

    p1_x .+= p2_y

    # p2_y: temporary variable to save memory
    p3_z .= p3 .- circshift!(p2_y, p3, [0, 0, 1])
    p3_z[:, :, end] .= -p3[:, :, end-1]
    p3_z[:, :,   1] .=  p3[:, :, 1]

    p1_x .+= p3_z
    return p1_x
end

function div!(divp::AbstractArray{T, 3}, p::AbstractArray{T, 3}) where {T<:AbstractFloat}
    p1, p2 = view(p,:,:,1), view(p,:,:,2)
    p1_x, p2_y = view(divp,:,:,1), view(divp,:,:,2)

    # p2_y is temp variable to save memory
    p1_x         .=  p1 - circshift!(p2_y, p1, [0, 1])
    p1_x[:, end] .= -p1[:, end-1]
    p1_x[:,   1] .=  p1[:, 1]

    circshift!(p2_y, p2, [1, 0])
    p2_y         .= p2 - p2_y
    p2_y[end, :] .= -p2[end-1, :]
    p2_y[1,   :] .=  p2[1, :]

    p1_x .+= p2_y
    return p1_x
end

"Compute backward divergence dim:[height, width]"
function div2d(p1, p2)
    p1_x         =  p1 - circshift(p1, [0, 1])
    p1_x[:, end] .= -p1[:, end-1]
    p1_x[:,   1] .=  p1[:, 1]

    p2_y = p2 - circshift(p2, [1, 0])
    p2_y[end, :] .= -p2[end-1, :]
    p2_y[1,   :] .=  p2[1, :]

    div_p = p1_x + p2_y
    return div_p
end

"Project p[H,W,2] to dual isonorm"
function proj_dual_iso!(p::AbstractArray{T, 3}, temp, weight) where {T<:AbstractFloat}
    "TODO: Avoid copying"
    p1, p2 = view(p,:,:,1), view(p,:,:,2)
    temp .= 1 ./ max.(1.0, sqrt.( p1 .^2 + p2 .^ 2 ) ./ (weight+1e-8) )
    p1 .*= temp
    p2 .*= temp
end

"Project p[H,W,Z,3] to dual isonorm"
function proj_dual_iso!(p::AbstractArray{T, 4}, temp, weight) where {T<:AbstractFloat}
    "TODO: Avoid copying"
    p1, p2, p3 = view(p,:,:,:,1), view(p,:,:,:,2), view(p,:,:,:,3)
    temp .= 1 ./ max.(1.0, sqrt.( p1 .^2 + p2 .^2 + p3 .^2 )./ (weight+1e-8))
    p1 .*= temp
    p2 .*= temp
    p3 .*= temp
end

function proj_dual_iso!(x::AbstractArray{T, 2}, weight::T) where {T<:AbstractFloat}
    x .= x-sign.(x) .* max.(abs.(x) .- weight, 0.0)
    return x
end

"Project l1 norm (soft thresholding)"
function proj_l1!(x::AbstractArray{T}, weight::T) where {T<:AbstractFloat}
    x .= sign.(x) .* max.(abs.(x) .- weight, 0.0)
end

"Prox to dual l1 norm = Project to unit norm ball"
function proj_dual_l1!(x::AbstractArray{T}, weight::T) where {T<:AbstractFloat}
    x ./= max.(1.0, abs.(x) / weight)
end


"""

Proximal operator of nuclear norm to x. y=prox_{s f}(x)


"""
function prox_nuclear!(y, x, stepsize)
    H, W, M, C = size(x) # Dim : 2 for 2D, 3 for 3D
    x_HxWxCx2 = PermutedDimsArray(x, [1,2,4,3])
    x_ = reshape(x_HxWxCx2, H*W, C, 2)
    y_ = reshape(y, H*W, 2, C)

    EPS = eps()

    x_svd = mapslices(svd!, x_, dims=[2,3])
    # x_svd = dropdims(x_svd_, dims=4)
    for (hw, svd_slice) in enumerate(x_svd)
        for i=1:2
            # svd_slice.S[i] svd_slice.S .- stepsize
            if (svd_slice.S[i] > EPS)
                svd_slice.S[i] = max(svd_slice.S[i] - stepsize, 0.0) / svd_slice.S[i]
            else
                svd_slice.S[i] = max(svd_slice.S[i] - stepsize, 0.0)
            end
        end
        x_hw = view(x_, hw, :, :) # [C x 2]
        y_hw = view(y_, hw, :, :) # [2 x C]

        svd_slice.Vt[1,1] *= svd_slice.S[1]
        svd_slice.Vt[2,2] *= svd_slice.S[2]

        mul!(x_hw, x_hw, svd_slice.Vt')
        mul!(x_hw, x_hw, svd_slice.Vt)

        y_hw[1,1] = x_hw[1,1]
        y_hw[2,2] = x_hw[2,2]
        y_hw[2,1] = x_hw[1,2]
        y_hw[1,2] = x_hw[2,1]
    end
    return y
end

end

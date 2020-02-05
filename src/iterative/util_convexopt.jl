# Copyright (c) Jakeoung Koo

module util_convexopt

export LinOp, compute_opnorm, proj_dual_iso!

# See http://www.ima.umn.edu/materials/2019-2020/SW10.14-18.19/28302/talk.pdf
struct LinOp
    forward::Function
    adjoint::Function
end
Base.:(*)(A::LinOp, x) = A.forward(x) # for A*x
Base.adjoint(A::LinOp) = LinOp(A.adjoint, A.forward) # for A'

D_  = f -> grad(f)
Dt_ = u -> -div(u)

"D: gradient operator"
D = LinOp(D_, Dt_)

export D

"Compute opeartor norm based on the Power method"
function compute_opnorm(A, niter=3)
    lamb = 0
    x = rand(size(A,2))

    for i=1:niter
        Ax = A*x # A'*A*x is very slow (WHY?) Todo
        x_next = A'*Ax
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

function grad(u::Array{T, 3}) where {T<:AbstractFloat}
    ux = circshift(u, [0, -1, 0]) - u
    uy = circshift(u, [-1, 0, 0]) - u
    uz = circshift(u, [0, 0, -1]) - u
    ux[:, end,   :] .= 0.0
    uy[end, :,   :] .= 0.0
    uz[:,   :, end] .= 0.0
    
    return cat(ux, uy, uz, dims=3)
end

function div(p::Array{T, 3}) where {T<:AbstractFloat}
    if size(p, 3) == 2
        return div2d(p[:,:,1], p[:,:,2])
    elseif size(p, 3) == 3
        return div3d(p[:,:,1], p[:,:,2], p[:,:,3])
    end
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

"Compute backward divergence dim:[height, width, slice]"
function div3d(p1, p2, p3)
    p1_x         =  p1 - circshift(p1, [0, 1, 0])
    p1_x[:, end, :] .= -p1[:, end-1, :]
    p1_x[:,   1, :] .=  p1[:, 1, :]
    
    p2_y = p2 - circshift(p2, [1, 0, 0])
    p2_y[end, :, :] .= -p2[end-1, :, :]
    p2_y[1,   :, :] .=  p2[1, :, :]

    p3_z = p3 - circshift(p3, [0, 0, 1])
    p3_z[end, :, :] .= -p3[:, :, end-1]
    p3_z[1,   :, :] .=  p3[:, :, 1]
    
    div_p = p1_x + p2_y
    return div_p
end

"Project p[H,W,2] to dual isonorm"
function proj_dual_iso!(p, weight)
    norms = sqrt.(p[:,:,1].^2 + p[:,:,2].^2)
    p[:,:,1] ./= max.(1.0, norms ./ (weight+1e-8))
    p[:,:,2] ./= max.(1.0, norms ./ (weight+1e-8))
end

"Project l1 norm"
function proj_l1(x, weight)
    return sign(x) .* max.(abs.(x), 0.0)
end


end

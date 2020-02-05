@doc raw"""
    recon2d_tv_primaldual(A, b, u0, niter, w_tv, c=1.0)

Reconstruct a 2d image by total variation model using Primal Dual optimization method

# Args
A : Forward opeartor
b : Projection data 
u0: Initial guess of image
niter: number of iterations

u <- u + CA'R(b - Au)
"""
function recon2d_sirt(A, b::Array{R, 1}, u0::Array{R, 2}, niter::Int; min_const=nothing, max_const=nothing) where {R <: AbstractFloat}
    R_mx1 = zeros(size(A, 1))
    C_nx1 = zeros(size(A, 2))

    sum_row = A * ones( size(A, 2) )
    EPS = eps(R)
    R_mx1[sum_row .> EPS] .= 1 ./ (sum_row[sum_row .> EPS])

    sum_col = A' * ones( size(A, 1) )
    C_nx1[sum_col .> EPS] .= 1 ./ (sum_col[sum_col .> EPS])

    u = vec(deepcopy(u0))

    for it = 1:niter
        # don't know why, but one line is too slow
        r = b .- A*u
        t1 = R_mx1 .* r
        bp = A' * t1
        u .+= C_nx1 .* bp

        # u = A' * r
        
        if min_const !== nothing
            u .= max.(u, min_const)
        end
        if max_const !== nothing
            u .= min.(u, max_const)
        end
        if it % 10 == 0
            residual = sum(abs.(r)) / length(r)
            println("$it residual: $residual")
        end
    end

    return reshape(u, size(u0))
end
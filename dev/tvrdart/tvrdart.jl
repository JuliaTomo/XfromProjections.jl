function _recon2d_slices_tvrdart!(u, A, b0, niter, w_tv, sigmas, tau)
    At = sparse(A')
    H, W = size(u)
    
    b = vec(b0)

    
    return u
end

function obj_param(A, p, u, w_tv, epsilon)
    
end

function der_param()

end

function _estimate_param(A, p, u, param0, w_tv, epsilon)
    obj = obj_param(A, p, u, w_tv, epsilon)
    objder = der_param()
    # residual = minimize()..

    param = res.x
    return param
end

"""
    recon2d_slices_tvrdart!(u::Array{T, 2}, A, b, niter::Int, w_tv::T, c=10.0)

Reconstruct 3D slice by slice by TVR-DART

# Args
u : Initial guess of image
A : Forward opeartor
b : Projection data (2 dimension or 1 dimension)
niter: number of iterations (<50)
w_tv: weight for TV term
"""
function recon2d_slices_tvrdart!(u::Array{T, 3}, A, b, niter::Int, w_tv::T, nmaterials, epsilon=1e-4, K=4) where {T <: AbstractFloat}
    
    tau = c / sum(ops_norm)
    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")

    recon2d_slices_sirt!(u, A, b, 100)
    u_max = maximum(u)
    b .= b ./ u_max
    u .= u ./ u_max

    # initial params
    K = 4*ones(nmaterials-1)
    gv = LinRange(0, 1, nmaterials)

    params = [gv[2:], K]
    
    # estimate optimal params by small subset 
    sum(u, dims=[1,3])

    # solve gray values and thresholds
    yhalf = Int(floor(size(u,2)/2))
    yrange = [ yhalf, yhalf+2 ]
    u_slice = u[:, yhalf:yhalf+2, :]
    data_slice = b[:, :, yhalf:yhalf+2]
    p_slice = reshape(data_slice)
    grays = range(0, stop=1, length=nmaterials)[2:end-1]
    param0 = [grays[2:end]..., w_tv]
    param = _estimate_param(A, p_slice, u_slice, param0, w_tv, epsilon)
    
    return _recon2d_slices_tvrdart!(u, A, b, niter, w_tv, param)
end

def gv2param(gv,K):
    param = np.hstack((gv[1:],K))
    return param;

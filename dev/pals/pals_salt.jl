using ParamLevelSet
using jInv.Mesh
using SparseArrays
using LinearAlgebra

function heavi(x, type="global", eps=0.1, thr=0.0) # [h,d] = 

    x .= x - thr;
    
    if type == "global"
        h = 0.5 * (1 + 2/pi*atan(pi* x /eps))
        d = 1.0 / (eps*((x .^ 2 * pi^2) / eps^2 + 1))

    elseif type == "compact"
        # compact
        # h = 0*x;
        # d = 0*x;
        # id = find((x < epsi) & (x > -epsi));
        # h(id) = 0.5*(1 + x(id)/epsi + 1/pi*sin(pi*x(id)/epsi));
        # h(x >= epsi) = 1;
        # h(x <=-epsi) = 0;
        # d(id) = 0.5*(1/epsi)*(1 + cos(pi*x(id)/epsi))
    end

    return h, d
end

function recon2d_slices_pals!(u::Array{T, 3}, A, p, niter::Int=50, nbasis::Int=4) where {T <: AbstractFloat}

    global Jt, J0, mesh, m, Hessian, grad, r

    mesh, m = init_pals(size(u), nbasis)
    sigmaH = getDefaultHeaviside();
    phat = similar(p)
    r = similar(p)
    At = A'
    stepsize = 0.01
    nslice = size(p,2)

    HW = size(u,1)*size(u,2)
    szpslice = size(p,1)*size(p,3)
    J = spzeros(prod(size(p)), nbasis*5) # A * J0

    for i=1:niter
        # u0, JBuilder = eval_f_grad(mesh, m; computeJacobian=1, sigma=sigmaH)
        u0, JBuilder = MeshFreeParamLevelSetModelFunc(mesh, m; computeJacobian = 1)

        J0 = getSparseMatrix(JBuilder)
    
        # construct J
        Threads.@threads for ss=1:nslice
            J[szpslice*(ss-1)+1:szpslice*ss, :] .= A * J0[HW*(ss-1)+1:HW*ss, :]
            
            u_slice = view(u, :, :, ss)
            Av = A*vec(u_slice)
            
            # phat_slice = reshape(Av, size(p,1), size(p,3))
            p_vec_ = reshape(Av, size(p,1), size(p,3))
            @views r[:,ss,:] = p_vec_ - p[:,ss,:]
        end
    
        Jt = J'
        Hessian = Jt*J
        Hinv = pinv(Matrix(Hessian))
        grad = Jt * vec(r)
        m .-= stepsize * Hinv * grad

        fres = sum(r .^ 2) / length(r)
        @show fres, m
        # println("residual: $fres, $m")
    end
    
    # u0, JBuilder = eval_f_grad(mesh,m;computeJacobian=1, sigma=sigmaH)
    u0, JBuilder = MeshFreeParamLevelSetModelFunc(mesh, m; computeJacobian = 1)
    u .= reshape(u0, size(u))
    return u, m
end
using .util_convexopt

is_initiated = false

function __init__()
    is_initiated = true
    py"""
    # reference: https://github.com/pathak22/pyflow.git
    import numpy as np
    import pyflow

    #find flow from img2 to img1
    def py_flow(img1,img2,alpha=0.5, ratio=0.75, minWidth=20, nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30, colType=1):
        img1 = np.array(img1)
        img2 = np.array(img2)
        height, width = img1.shape

        im1 = np.array(img1).reshape(height,width,1).copy(order='C')
        im2 = np.array(img2).reshape(height,width,1).copy(order='C')

        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)

        return u, v, im2W
    """
end

function get_flows(u::Array{T,3}, alpha=0.5, ratio=0.75, minWidth=20, nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30, colType=1)::Array{T,4} where {T<:AbstractFloat}
    #ref: https://github.com/pathak22/pyflow.git
    @suppress begin
        #HACK!
        if !is_initiated
            __init__()
        end
        height, width, frames = size(u)
        result = zeros(height, width, 2, frames)
        for i in 1:frames-1
            flow_x, flow_y, im1Warped = py"py_flow"(u[:,:,i+1],u[:,:,i], alpha=alpha, ratio=ratio, minWidth=minWidth, nOuterFPIterations=nOuterFPIterations, nInnerFPIterations=nInnerFPIterations, nSORIterations=nSORIterations, colType=colType)
            result[:,:,1,i] = flow_x
            result[:,:,2,i] = flow_y
        end
        return result
    end
end

function sub2ind(array_shape, rows, cols)
    out = (cols.-1).*array_shape[1] .+ rows
    return convert.(Int64, out)
end

function compute_warping_operator(flow)
    # ref: https://github.com/HendrikMuenster/flexBox/blob/79abc7285703911cca2653434cca5bcefc79c722/operators/generateMatrix/warpingOperator.m
    """
    u: gray image
    v: flow [H, W, 2]
    """
    H, W, dims = size(flow)
    shape = (H,W)
    XX, YY = repeat(collect(1:W)', H, 1), repeat(collect(1:H), 1, W)

    targetPoint2Mat = XX + flow[:,:,1]
    targetPoint1Mat = YY + flow[:,:,2]

    x1 = floor.(targetPoint1Mat).-1
    x2 = x1 .+ 1
    x3 = x1 .+ 2
    x4 = x1 .+ 3

    y1 = floor.(targetPoint2Mat).-1
    y2 = y1 .+ 1
    y3 = y1 .+ 2
    y4 = y1 .+ 3

    v2 = targetPoint1Mat .- x2
    v1 = targetPoint2Mat .- y2

    indicator = ones(H,W)
    indicator = (x1 .> 0).*indicator
    indicator = (x4 .<= W).*indicator
    indicator = (y1 .> 0).*indicator
    indicator = (y4 .<= H).*indicator

    indicator = indicator .> 0
    v1 = v1[indicator]
    v2 = v2[indicator]

    idxList = sub2ind(shape, YY, XX)
    idxList_ = collect(Iterators.flatten(idxList))

    #find the indices
    list_J  = vcat(
        idxList_[sub2ind(shape, x2[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x2[indicator], y3[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y3[indicator])])

    list_J = collect(Iterators.flatten(list_J))

    list_val  = ( (1 .-v1) .* (1 .-v2))
    list_val = vcat(list_val,(v2 .* (1 .-v1)))
    list_val = vcat(list_val,(v1 .* (1 .-v2)))
    list_val = vcat(list_val,(v1 .* v2))

    list_val = collect(Iterators.flatten(list_val))

    list_I = idxList[indicator]
    list_I =collect(Iterators.flatten(repeat(list_I,1,4)))

    W_mat = dropzeros!(sparse(list_I, list_J, list_val, H*W, H*W))

    return LinearOperator(W_mat)
end


function _recon2d_tv_primaldual_flow(As,A_norm,∇_norm, W_list,u0s,bs,w_tv,w_flow,c,niter,p1,p2,p3, mask)
    #A variational reconstruction method for undersampled dynamic x-ray tomography based on physical motion models (Burger)#
    height, width, frames = size(u0s)

    u = u0s
    u_prev = similar(u)
    ubar = deepcopy(u)

    p_adjoint  = similar(p3)
    p3_adjoint = similar(p3)#zeros(height*width, frames)
    p_adjoint_prev = similar(p3)
    W_norm = maximum(map(W-> compute_opnorm(W), W_list))
    opsnorm = [A_norm, ∇_norm, W_norm]
    tau = c /sum(opsnorm)

    data_term = similar(bs)
    u_grad = similar(p2)
    u_temp = similar(u)
    divp2 = similar(p2)

    flow_term = similar(u)

    sigmas = map(n -> 1.0/(n*c), opsnorm)

    @info sigmas
    it = 1
    primal_gap = 100
    tol = 0.001
    while it < niter && primal_gap > tol
        u_prev .= u
        p_adjoint_prev .= p_adjoint
        for t=frames:-1:1
            A = As[t]
            #dual
            @views data_term[:,t] .= mul!(data_term[:,t], A, vec(ubar[:,:,t])) .- bs[:,t]
            @views p1[:,t] .= (p1[:,t] .+ sigmas[1] * data_term[:,t]) ./ (sigmas[1] + 1.0)# l2 norm
            @views mul!( view(p_adjoint[:,:,t], :), A', p1[:,t]) #projection?

            @views util_convexopt.grad!(u_grad[:,:,:,t], ubar[:,:,t])
            @views p2[:,:,:,t] .+= sigmas[2] .* u_grad[:,:,:,t]
            @views util_convexopt.proj_dual_iso!(p2[:,:,:,t], u_temp[:,:,t], w_tv)
            @views p_adjoint[:,:,t] .-= util_convexopt.div!(divp2[:,:,:,t], p2[:,:,:,t])

            if t < frames
               @views mul!(view(flow_term[:,:,t], :), W_list[t], vec(ubar[:,:,t]))
               @views flow_term[:,:,t].-=ubar[:,:,t+1]
               @views p3[:,:,t] .= (p3[:,:,t] .+ sigmas[3] * flow_term[:,:,t])
               @views util_convexopt.proj_dual_iso!(p3[:,:,t], w_flow)

               @views mul!(view(p3_adjoint[:,:,t], :), W_list[t]', vec(p3[:,:,t]))
               @views p_adjoint[:,:,t] .+= p3_adjoint[:,:,t]
            end
        end
        u .-= tau * p_adjoint
        # primal update
        u .= max.(u.-tau .* p_adjoint, 0.0) # positivity constraint
        u .*=mask

        # acceleration
        ubar .= 2 .* u .- u_prev

        if it % 50 == 0
            #du = u_prev - u
            primal_gap = sum(abs.(-p_adjoint+p_adjoint_prev + (u_prev-u)/tau)) / length(p_adjoint)
            #@info "primal gap:" primal_gap
            @info it primal_gap
        end
        it = it+1
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
function recon2d_tv_primaldual_flow(A_list, bs::Array{R, 2}, u0s::Array{R, 3}, niter1::Int, niter2::Int, w_tv::Float64, w_flow::Float64, c, mask, alpha) where {R <: AbstractFloat}

    height,width,frames = size(u0s)
    v = zeros(height, width, 2, frames)
    u = u0s
    shape = (height*width, frames)
    A_norm = maximum(map(A->compute_opnorm(A), A_list))#compute_opnorm_block(A_list, shape)
    N = height #only ok for height=width
    Dfd = spdiagm(0 => -ones(N-1), 1=>ones(N-1))
    ∇ = kron(Dfd, sparse(I,N,N)) + kron(Dfd, sparse(I.*im, N, N))
    ∇_norm = abs(compute_opnorm(∇))

    p1 = zeros(size(bs))
    p2 = zeros(height, width, 2,frames)
    p3 = zeros(height, width, frames)
    tol = 0.001
    i = 1
    rmain = 100
    while i < niter1 && rmain > tol
        u_prev = u
        v_prev = v
        W_list = mapslices(f -> compute_warping_operator(f), v,dims=[1,2,3])

        #W_norm = map(W)=#compute_opnorm_block(W_list, shape)
        u = _recon2d_tv_primaldual_flow(A_list,A_norm,∇_norm, W_list,u,bs,w_tv,w_flow,c,niter2,p1,p2,p3, mask)

        v = get_flows(u, alpha)
        rmain = norm(u-u_prev)+norm(v-v_prev)
        i = i+1
        @info i rmain
    end
    return u
end

using LinearAlgebra

function proj_dual_iso2d!(p, temp, weight)
    p1, p2 = view(p,:,:,1), view(p,:,:,2)
    temp .= 1 ./ max.(1.0, sqrt.( p1 .^2 + p2 .^ 2 ) ./ (weight+1e-8) )
    p1 .*= temp
    p2 .*= temp
end

"Project to dual l1 norm"
function proj_dual_l1!(x, weight)
    x ./= max.(1.0, abs.(x) / weight)
end

function div2d!(divp, p)
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

function grad2d!(du, u) where {T<:AbstractFloat}
    du1, du2 = view(du, :, :, 1), view(du, :, :, 2)
    du1 .= circshift!(du1, u, [0, -1]) .- u
    du2 .= circshift!(du2, u, [-1, 0]) .- u
    du1[:, end] .= 0.0
    du2[end, :] .= 0.0
end


"Compute opeartor norm based on the Power method"
function compute_opnorm_block(As,shape, niter=3)
    λ = 0
    size,count = shape
    x = rand(size*count)

    for i=1:niter
        x_next = zeros(size*count)
        for j=1:count
            A = As[j]
            Ax = A*x[((j-1)*size)+1:((j-1)*size)+size]
            x_next_j = A'*Ax
            x_next[((j-1)*size)+1:((j-1)*size)+size] = x_next_j
        end
        λ_prev = λ
        λ = sqrt(sum(x_next.^2))

        if (abs(λ_prev-λ) < 1e-9)
            println("break at iteration: ", i)
            break
        end
        x = x_next / λ
    end
    return sqrt(λ)
end

function _recon2d_tv_primaldual_flow(As,A_norm,W_list,W_norm,u0s,bs,w_tv,w_flow,c,niter,p1,p2,p3)
    #A variational reconstruction method for undersampled dynamic x-ray tomography based on physical motion models (Burger)#
    height, width, frames = size(u0s)

    u = u0s
    u_prev = similar(u)
    ubar = deepcopy(u)

    p_adjoint  = similar(p3)
    p3_adjoint = similar(p3)
    p_adjoint_prev = similar(p3)
    opsnorm = [A_norm, sqrt(8), W_norm]
    tau = c /sum(opsnorm)

    data1 = similar(bs)
    u_grad = similar(p2)
    u_temp = similar(u)
    divp2 = similar(p2)

    Wus = similar(u)

    sigmas = map(n -> 1.0/(n*c), opsnorm)
    #@info sigmas

    for it=1:niter
        u_prev .= u
        p_adjoint_prev .= p_adjoint
        Threads.@threads for t=1:frames
            A = As[t]
            #sigmas = map(n-> 1.0/(n*c), opsnorm)

            @views data1[:,t] .= mul!(data1[:,t], A, vec(ubar[:,:,t])) .- bs[:,t]
            @views p1[:,t] .= (p1[:,t] .+ sigmas[1] * data1[:,t]) / (sigmas[1] + 1.0)
            @views mul!( view(p_adjoint[:,:,t], :), A', p1[:,t])

            @views grad2d!(u_grad[:,:,:,t], ubar[:,:,t])
            @views p2[:,:,:,t] .+= sigmas[2] .* u_grad[:,:,:,t]
            @views proj_dual_iso2d!(p2[:,:,:,t], u_temp[:,:,t], w_tv)

            @views p_adjoint[:,:,t] .-= div2d!(divp2[:,:,:,t], p2[:,:,:,t])

            if t < frames
                _p3 = view(p3, :, :, t)
                Wu = view(Wus, :, :, t)
                Wuv = vec(Wu)
                @views Wuv .= mul!(Wuv, W_list[t], vec(ubar[:,:,t+1])) .- vec(ubar[:,:,t])

                @views _p3 .+= sigmas[3] .* Wu
                proj_dual_l1!(_p3, w_flow)
                p3_adj = view(p3_adjoint[:,:,t], :)

                mul!(p3_adj, W_list[t]', vec(_p3))
                # @views p_adjoint[:,:,t+1] .+= p3_adjoint[:,:,t] # race condition
                @views p_adjoint[:,:,t] .-= _p3
            end
        end

        # add p3_adjoint to prevent race condition
        @views p_adjoint[:,:,2:end] .+= p3_adjoint[:,:,1:end-1]

        # primal update
        u .-= tau * p_adjoint
        u .= max.(u, 0.0) # positivity constraint

        # acceleration
        ubar .= 2*u .- u_prev

        if it % 50 == 0
            #du = u_prev - u
            primal_gap = sum(abs.(-p_adjoint+p_adjoint_prev + (u_prev-u)/tau)) / length(p_adjoint)
            #@info "primal gap:" primal_gap
            @info it primal_gap
        end
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
function recon2d_tv_primaldual_flow(A_list, bs::Array{R, 2}, u0s::Array{R, 3}, niter1::Int, niter2::Int, w_tv::R, w_flow::R, c=1.0) where {R <: AbstractFloat}

    height,width,frames = size(u0s)
    v = zeros(height, width, 2, frames)
    u = u0s
    shape = (height*width, frames)
    A_norm = compute_opnorm_block(A_list, shape)

    p1 = zeros(size(bs))
    p2 = zeros(height, width, 2,frames)
    p3 = zeros(height, width, frames)
    for i=1:niter1
        u_prev = u
        v_prev = v
        W_list = mapslices(f -> compute_warping_operator(f), v,dims=[1,2,3])
        # check if W_list is correct
        println(sum(abs.(W_list[1]*vec(u[:,:,2]) - vec(u[:,:,1]))))
        @views @info "check if W_list is correct. data term after and before warping" sum(abs.(W_list[1]*vec(u[:,:,2]) - vec(u[:,:,1])))/(height*width) sum(abs.(u[:,:,2] - u[:,:,1]))/(height*width)

        W_norm = compute_opnorm_block(W_list, shape)
        u = _recon2d_tv_primaldual_flow(A_list,A_norm,W_list,W_norm,u,bs,w_tv,w_flow,c,niter2,p1,p2,p3)
        v = get_flows(u)
    end
    return u
end

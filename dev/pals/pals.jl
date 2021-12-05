using ParamLevelSet
using jInv.Mesh
using SparseArrays
using LinearAlgebra

# SpMatBuilder = ParamLevelSet.SpMatBuilder

# function getAlpha(k::Int64,m::Vector{Float64},numParamOfRBF::Int64 = 5)
# 	return m[numParamOfRBF*(k-1) + 1];
# end

# function getBeta(k::Int64,m::Vector{Float64})
# 	return m[5*(k-1) + 2];
# end

# function getX(k::Int64,m::Vector{Float64},numParamOfRBF::Int64 = 5)
# 	return m[numParamOfRBF*(k-1).+((numParamOfRBF-2):numParamOfRBF)];
# end

# function getXt(k::Int64,m::Vector{Float64},numParamOfRBF::Int64 = 5)
# 	return (m[numParamOfRBF*(k-1)+numParamOfRBF-2],m[numParamOfRBF*(k-1)+numParamOfRBF-1],m[numParamOfRBF*(k-1)+numParamOfRBF]);
# end

# function setX(k::Int64,m::Vector{Float64},x::Array)
# 	m[5*(k-1)+(3:5)] = x;
# end


# function computeFunc!(m::Vector{Float64},k::Int64,Xc,h::Vector{Float64},n::Vector{Int64},
#     u::Vector{Float64},xmin::Vector{Float64})
# (x1,x2,x3) = getXt(k,m);
# i = round.(Int64,(getX(k,m) - xmin)./h .+ 0.5);
# # ## here we take a reasonable box around the bf.
# # ## The constant depends on the type of RBF that we take.
# beta = getBeta(k,m);
# betaSq = beta^2;
# thres = 0.81/betaSq;
# boxL = ceil.(Int64,0.9./(h*abs(beta)));
# imax = min.(i + boxL,n);
# imin = max.(i - boxL,[1;1;1]);
# alpha = getAlpha(k,m);

# @inbounds for l = imin[3]:imax[3]
# @inbounds iishift3 = (l-1)*n[1]*n[2];
# @inbounds for j = imin[2]:imax[2]
#    @inbounds iishift2 = iishift3 + (j-1)*n[1];
#    @inbounds for q = imin[1]:imax[1]
#        ii = iishift2 + q;
#        @inbounds y = Xc[ii,1]-x1; y*=y;
#        nX = y;
#        @inbounds y = Xc[ii,2]-x2; y*=y;
#        nX+=y;
#        @inbounds y = Xc[ii,3]-x3; y*=y;
#        nX+=y;
#        if (nX <= thres) # 0.77~0.875^2
#            nX*=betaSq;
#            argii = radiust(nX);
#            argii = psi1(argii);
#            argii *= alpha;
#            @inbounds u[ii] += argii;
#        end
#    end
# end
# end
# end


# function updateJacobian!(k::Int64,m::Vector{Float64},dsu::Vector{Float64},h::Vector{Float64},n::Vector{Int64},Jbuilder::SpMatBuilder,u::Vector{Float64},iifunc::Function,Xc,xmin::Vector{Float64})
# (x1,x2,x3) = getXt(k,m);
# alpha = getAlpha(k,m);
# beta = getBeta(k,m);
# betaSQ = beta*beta;
# invBeta = (1.0/beta);
# thres = 0.81/betaSQ;


# alphaBetaSq = alpha*betaSQ;
# offset = convert(Int64,(k-1)*5 + 1);
# md = 1e-3*maximum(abs.(dsu));
# boxL = ceil.(Int64,0.9./(h*abs(beta)));
# i = round.(Int64,([x1;x2;x3] - xmin)./h .+ 0.5);
# imax = min.(i + boxL,n);
# imin = max.(i - boxL,[1;1;1]);

# @inbounds for l = imin[3]:imax[3]
# @inbounds iishift3 = (l-1)*n[1]*n[2];
# @inbounds for j = imin[2]:imax[2]
#    @inbounds iishift2 = iishift3 + (j-1)*n[1];
#    @inbounds for q = imin[1]:imax[1]
#        ii = iishift2 + q;
#        temp = dsu[ii];
#        if temp >= md
#            @inbounds y1 = x1 - Xc[ii,1]; nX =y1*y1;
#            @inbounds y2 = x2 - Xc[ii,2]; nX+=y2*y2;
#            @inbounds y3 = x3 - Xc[ii,3]; nX+=y3*y3;
#            if (nX <= thres) # 0.77~0.875^2
#                radii = radiust(nX*betaSQ);
#                psi,dpsi = dpsi1_t(radii);
#                psi*=temp;
#                temp*= alphaBetaSq;
#                temp/= radii;
#                temp*=dpsi;
#                nX *= temp;
#                nX *= invBeta;
#                y1*=temp; y2*=temp; y3*=temp;
#                ii = iifunc(ii);
#                setNext!(Jbuilder,ii,offset,psi);
#                setNext!(Jbuilder,ii,offset+1,nX);
#                setNext!(Jbuilder,ii,offset+2,y1);
#                setNext!(Jbuilder,ii,offset+3,y2);
#                setNext!(Jbuilder,ii,offset+4,y3);
#            end
#        end
#    end
# end
# end
# end

# function MeshFreeParamLevelSetModelFunc(Mesh::RegularMesh, m::Vector; computeJacobian = 1, bf::Int64 = 1,
#     sigma::Function = (m,n)->(n[:] .= 1.0),
#     Xc = convert(Array{Float64,2}, getCellCenteredGrid(Mesh)),u::Vector = zeros(size(Xc,1)),dsu::Vector = zeros(size(Xc,1)),
#     Jbuilder::SpMatBuilder{Int64,Float64} = getSpMatBuilder(Int64,Float64,size(Xc,1),length(m),computeJacobian*10*size(Xc,1)),
#     iifunc::Function=identity, numParamOfRBF::Int64 = 5,tree = BallTree(Matrix(Xc')))

#     if length(u) != size(Xc,1)
#         error("preallocated u is of wrong size");
#     end

#     u[:] .= 0.0;
#     reset!(Jbuilder);
#     nRBFs = div(length(m),numParamOfRBF);
#     ## in run 1 we calculate u. in run 2 we calculate J after we know the derivative of the heaviside.
#     beta_arr = Array{Float64}(undef,nRBFs);

#    for k=1:nRBFs
#        MeshFreeComputeFunc!(m,k,u,Xc,tree);
#    end

#    sigma(u,dsu);
#     if computeJacobian == 1
#        for k=1:nRBFs
#            MeshFreeUpdateJacobian!(k,m,dsu,Jbuilder,u,iifunc,Xc,tree);
#        end
#     end
#     return u,Jbuilder;
# end

# function eval_f_grad()
#     sigma::Function = (m,n)->(n[:] .= 1.0),
#     Xc = convert(Array{Float64,2}, getCellCenteredGrid(Mesh)),u::Vector = zeros(size(Xc,1)),dsu::Vector = zeros(size(Xc,1)),
#     # Jbuilder::SpMatBuilder{Int64,Float64} = getSpMatBuilder(Int64,Float64,size(Xc,1),length(m),computeJacobian*10*size(Xc,1)),
#     # iifunc::Function=identity, numParamOfRBF::Int64 = 5,tree = BallTree(Matrix(Xc')))

#     if length(u) != size(Xc,1)
#         error("preallocated u is of wrong size");
#     end

#     u[:] .= 0.0;
#     reset!(Jbuilder);
#     nRBFs = div(length(m),numParamOfRBF);
#     ## in run 1 we calculate u. in run 2 we calculate J after we know the derivative of the heaviside.
#     beta_arr = Array{Float64}(undef,nRBFs);

#    for k=1:nRBFs
#        MeshFreeComputeFunc!(m,k,u,Xc,tree);
#    end

#    sigma(u,dsu);
#     if computeJacobian == 1
#        for k=1:nRBFs
#            MeshFreeUpdateJacobian!(k,m,dsu,Jbuilder,u,iifunc,Xc,tree);
#        end
#     end
#     return u,Jbuilder;
# end

function init_pals(vol_size, nbasis=4)
	half = vol_size ./ 2
    # Mesh = getRegularMesh([-half[1];half[1];-half[2];half[2];-half[3];half[3]],[vol_size[1],vol_size[2],vol_size[3]]);
    Mesh = getRegularMesh([1;vol_size[1];1;vol_size[2];1;vol_size[3]],[vol_size[1],vol_size[2],vol_size[3]]);

	alpha = [1.5;2.5;-2.0;-1.0];
	beta = [2.5;2.0;-1.5;2.5]*0.001
	Xs = [0.5 0.5 0.5; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0] .* vol_size[1] .* 0.3
	m = wrapTheta(alpha,beta,Xs);

	return Mesh, m
end

function recon2d_slices_pals!(u::Array{T, 3}, A, p, niter::Int=1, nbasis::Int=4) where {T <: AbstractFloat}

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
        for ss=1:nslice
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

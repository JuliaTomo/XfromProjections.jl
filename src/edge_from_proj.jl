"""
Extract blobs from projections
"""


"""
    peak_extreme(scale_space, sz_nbd)

# Args
- `scale_space:Array`: scale space [nscale, H, W]
"""
function peak_extreme(scale_space, sz_nbd)
    # Ref: https://github.com/scikit-image/scikit-image/blob/b7a1fde540d90d6e8009764abeec3e89c8b76928/skimage/feature/peak.py#L24
    nscale, H, W = size(scale_space)
    
end

"radon transform of LoG"
# $$e_{\sigma}(t)=\left(\frac{1}{\sqrt{2 \pi} \sigma^{5}}\right)\left(t^{2}-\sigma^{2}\right) \exp ^{-t^{2} / 2 \sigma^{2}}$$
# note that we consider 3D

_radon_log(t, sigma; z0) = (t.^2 .- sigma^2 .+ z0^2) .* exp.(-(t.^2 + z0^2 ) / ( 2*sigma^2 ))

"Compute Radon transform of normalized Laplacian of Gaussian in 3D"
function radon_log(q, sigma; z0=0)
    
    nangles, detcount = size(q)
    p_log = zeros(nangles, detcount)
    tt = (-detcount/2+0.5):1.0:(detcount/2-0.5)

    for ang in 1:nangles
        for (i, t) in enumerate(tt)
            dotp = q[ang, :] .* _radon_log.(t .- tt, sigma, z0=z0) ./ ( sqrt(2*pi) * sigma^3 )

            p_log[ang, i] = sum(dotp)
        end
    end
    
    return p_log
end

"(Not implemenetd yet) Compute scale space from a projection with different scales"
function ss_from_proj(q, A; nsigma=8, mult_sigma=1, sigma0=1.6/sqrt(2))
    detcount, nangles = size(q)
    
    k = sqrt(2)
#     sigma0 = 1.6 / k
    
    sigmas = zeros(nsigma)
    for i=1:nsigma
        sigmas[i] = i * k * mult_sigma * sigma0
    end

    p_logs = zeros(nsigma, nangles, detcount)


    # f*g (t) = \sum_x f(x) g(t-x)
    Threads.@threads for isigma in 1:nsigma
        sigma = sigmas[isigma]
        p_logs[isigma, :, :] = radon_log(q, sigma=sigma)
    end
    
#     img_logs = zeros(nsigma, size(img, 1), size(img, 2))
#     img_logs[abs.(img_logs) .< 0.2] .= 0.0;

#     sum_logs = zeros(size(img))
#     for isigma in 1:nsigma
#         img_sigma = reshape(A' * vec(p_logs[isigma,:,:]), size(img))
#         img_logs[isigma,:,:] = img_sigma
#         sum_logs .+= img_sigma
#     end

#     maxima = findlocalmaxima(img_logs, 1:ndims(img_logs), true)
#     idx_sigma = 1
#     blobs = []
#     for x in maxima
#         if x[2] > 10 && x[2] < H-10 && x[3] > 10 && x[3] < W-10
#             sigma = sigmas[Int(x[1])]
#     #         print(sigma)
#             if x[1] == idx_sigma
#                 push!(blobs, BlobLoG(CartesianIndex(Base.tail(x.I)), sigmas[x[1]], img_logs[x]))
#             end
#         end
#     end
#     sort!(blobs, by = v -> v.amplitude, rev=true)

#     # # # choose maximum 10
#     blobs = blobs[1:div(end,2)]
    
    
# sum_logs = zeros(size(img))
# for isigma in 1:nsigma
#     img_sigma = reshape(A' * vec(p_logs[isigma,:,:]), size(img))
#     img_logs[isigma,:,:] = img_sigma
#     sum_logs .+= img_sigma
# end
    
    return p_logs, sigmas;
end
"radon transform of LoG"
# $$e_{\sigma}(t)=\left(\frac{1}{\sqrt{2 \pi} \sigma^{5}}\right)\left(t^{2}-\sigma^{2}\right) \exp ^{-t^{2} / 2 \sigma^{2}}$$
# note that we consider 3D

"Compute Radon transform of normalized Laplacian of Gaussian in 3D"
function radon_log(q, sigma; z0=0)
    _radon_log(t, ang; z0=0) = (t.^2 .- sigma^2 .+ z0^2) .* exp.(-(t.^2 + z0^2 ) / ( 2*sigma^2 ))

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

"""
    radon_filter(p, angles, fun_filter, filt_sz)

Radon transform of a filter slice by slice specified by a 1D fucntion `fun_filter(theta,angle)`
"""
function radon_filter(p::Array{T, 3}, angles::Array{T, 1}, fun_filter, szfilter::Int) where {T<:AbstractFloat}
    if !isodd(szfilter)
        println("! error: only support odd number of filter size, szfilter is increased by 1")
        szfilter += 1
    end
    
    nangles, nslice, detcount = size(p)

    halfsz = div(szfilter-1, 2)    
    tt = collect(-halfsz:1:halfsz)
    out = zeros(size(p))
    filt_ = zeros(2*halfsz+1)
    
    for ang=1:nangles
        filt_ .= fun_filter.(tt, angles[ang])
        for slice=1:nslice
            for i = 1:2*halfsz+1
                let i=i # needed for simd
                    fi = filt_[i]
                    @simd for j = halfsz+1:(detcount-halfsz)
                        @inbounds out[ang, slice, i+j-(halfsz+1)] += p[ang, slice, j] * fi
                    end
                end
            end
        end
    end
    return out
end

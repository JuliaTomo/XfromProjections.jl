using FFTW
using DSP

"make fourier filter size_padded is power of 2"
function make_filter(sz, filter_type="ramlak")
    # See Kak Chapter3 (61)
    
    tau = cat(Array(1:2:Int(sz/2-1)), Array(Int(sz/2-1):-2:0), dims=1);
    spatial_filter = zeros(sz)
    spatial_filter[1] = 0.25
    spatial_filter[2:2:end] .= -1.0 ./ (pi * tau).^2
    
    fourier_filter = 2 * real(fft(spatial_filter))
    
    if filter_type == "shepplogan"
        omega = pi * FFTW.fftfreq(sz)[2:end]
        fourier_filter[2:end] .*= sin.(omega) ./ omega
    elseif filter_type == "hamming"
        fourier_filter .*= fftshift(hamming(sz))    
    elseif filter_type == "hann"
        fourier_filter .*= fftshift(hanning(sz))        
    elseif filter_type == "kaiser"
        fourier_filter .*= fftshift(kaiser(sz, 3.0))
    elseif filter_type == "consine"
        fourier_filter .*= fftshift(consine(sz))
    end
    
    return fourier_filter
end

@doc raw"""
    function filter_proj_freq(p; filter_type)

Filter projections p by a given filter type (p: [detcount x nangles])

Test eq ``E=mc^2``

``\frac{n!}{k!(n - k)!} = \binom{n}{k}``

supported filter_type:
ramlak, shepplogan, hamming, hann
"""
function filter_proj(p; filter_type="ramlak")
    # See Kak ch3 (61)

    nangles, detcount = size(p)
    
    # compute next power of 2
    size_padded = Int(2^ceil(log(detcount)/log(2)))
    fourier_filter = make_filter(size_padded, filter_type)
    
    dpad = size_padded - detcount
    
    # pad data
    p_padded = zeros(nangles, size_padded)
    for i=1:nangles
        p_padded[i, 1:detcount] = p[i, :]
    end
    
    # filter projection in Fourier domain
    proj = fft(p_padded, 2) .* reshape(fourier_filter, 1, length(fourier_filter))
    pimg_ifft = real(ifft(proj, 2)[:, 1:detcount])
    return pimg_ifft / 2.0; # for matching with the original
end


"""
    filter_proj_hilbert

Filter projections based on Hilbert transform.

(p: [detcount x nangles])
"""
# $q(s, \theta)=\frac{\mathrm{d} p(s, \theta)}{\mathrm{d} s} * \frac{-1}{2 \pi^{2} s}$
function filter_proj_hilbert(p)
    qimg = zeros(size(p))
    for i in 1:size(qimg, 1)
        qimg[i, 2:(end-1)] = pimg[i, 3:end] - pimg[i, 1:(end-2)]
    #     qimg[end,i] = 0
        qimg[i,:] *= -1.0 / (2*pi)
        qimg[i,:] = real(hilbert(qimg[i,:]))
    end
    return qimg
end

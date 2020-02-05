using DSP
using FFTW

export make_freq_window, gridrec

"Make window function in frequency domain with 3x3 size for crop=1"
function make_freq_window(sz; crop=1, option1=3.0, window_type="kaiser")
    # Ref: https://github.com/miaosiSari/Regridding/blob/master/get_window.m
    
    if window_type == "kaiser"
        vec_ = DSP.kaiser(sz, option1)
        
    elseif window_type == "hanning"
        vec_ = DSP.hanning(sz)
    end
    
    mat = vec_ * vec_'
    wf = fftshift(fft(ifftshift(mat)))
    
    if Bool(sz & 1)
        wf_crop = wf[div(sz+1,2)-crop:div(sz+1,2)+crop, div(sz+1,2)-crop:div(sz+1,2)+crop]
    else
        wf_crop = wf[div(sz,2)-crop:div(sz+1,2)+crop, div(sz+1,2)-crop:div(sz+1,2)+crop]
    end
    
    return wf, wf_crop
end

"""
    gridrec (p)

Reconstruct based on [1] gridrec

[1] : Marone, F., Stampanoni, M., 2012. Regridding reconstruction algorithm
    for real-time tomographic imaging. Journal of Synchrotron Radiation
"""
function gridrec(p, angles)
    nangles, detcount = size(p)
    
    # in [1], the author says that minimum padding should be detcount-1
    if Bool(detcount & 1)
        padding = detcount - 1
    else
        padding = detcount
    end
    
    padding_half = div(padding,2)
    sz = detcount + padding
    p_pad = zeros(sz, nangles)
    for i=1:detcount
        p_pad[:, i+padding_half] = p[:, i]
    end
    P = fftshift(fft(ifftshift(p_pad), 1));
    
    window, kernel = make_freq_window(sz)
    sz_kernel = size(kf, 1)
    sz_hkernel = div(sz_kernel, 2)
    
    # make not-shifted spatial filter
    tau = cat(Array(1:2:Int(sz/2-1)), Array(Int(sz/2-1):-2:0), dims=1);
    spatial_filter = zeros(sz)
    spatial_filter[1] = 0.25
    spatial_filter[2:2:end] .= -1.0 ./ (pi * tau).^2
    
    ramlak = fftshift(fft(spatial_filter)) # even function
    
    Q = zeros(Complex, sz, sz)
    
    # compute Q(U,V) in (6)
    for m=1:nangles
        θ = angles[m]
        cosθ = cos(θ)
        sinθ = sin(θ)
        for i=1:sz
            tau = i - div(sz, 2)
            tc = tau * cosθ
            ts = tau * sinθ
            
            P_wθ = P[i, m]
            
            for h = -sz_hkernel:1:sz_hkernel
                for w = -sz_hkernel:1:sz_hkernel

                    U = Int(round(tc+div(sz,2)+w));
                    V = Int(round(ts+div(sz,2)+h));
                    
                    if U > sz
                        U -= sz
                    end
                    if U < 1
                        U += sz
                    end
                    if V > sz
                        V -= sz
                    end
                    if V < 1
                        V += sz
                    end
                    
                    C = kernel[w+sz_hkernel+1, h+sz_hkernel+1]
#                     C*P_wθ*W[i]
                    
#                     print("$U,$V,")
                    Q[U, V] += C * P_wθ * ramlak[i]
                                        
                end
            end
        end
    end
    
    dθ = angles[2]-angles[1]
    Q = Q * dθ ./ (sz*sz)
    q = fftshift(ifft(ifftshift(Q)))
    recon = real(q ./ window)
    recon[recon.>1] .= 1.0
    recon[recon.<1e-3] .= 0.0
    recon[recon.==NaN] .= 0.0
#     print(mean(recon))
    
    return recon, Q
end
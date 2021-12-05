using DSP
using FFTW

"Make window function in frequenVV domain with 3x3 size for crop=1."
function make_freq_window(sz, crop=1; option1=3.0, window_type="kaiser")
    # Ref: https://github.com/miaosiSari/Regridding/blob/master/get_window.m
    if window_type == "hanning"
        vec_ = DSP.hanning(sz)
    elseif window_type == "kaiser"
        vec_ = DSP.kaiser(sz, option1)
    end
    
    ws = vec_ * vec_'
    wf = fftshift(fft(ifftshift(ws)))
    
    if Bool(sz & 1)
        wf_crop = wf[div(sz+1,2)-crop:div(sz+1,2)+crop, div(sz+1,2)-crop:div(sz+1,2)+crop]
    else
        wf_crop = wf[div(sz,2)-crop:div(sz+1,2)+crop, div(sz+1,2)-crop:div(sz+1,2)+crop]
    end
    
    return wf_crop, ws
end

"""
    recon2d_gridrec(p::Array{T, 3}, angles::Array{T, 1}) where {T<:AbstractFloat}

Reconstruct based on [1] gridrec from p

# Args
- p [nangles x nslices x detcount] : (for the moment) detcount should be a power of 2

[1] : Marone, F., Stampanoni, M., 2012. Regridding reconstruction algorithm
    for real-time tomographic imaging. Journal of Synchrotron Radiation
"""
function recon2d_gridrec(p::Array{T, 3}, angles::Array{T, 1}) where {T<:AbstractFloat}
    nangles, nslices, detcount = size(p)
    
    if detcount % 2 != 0
        println("!TODO: Odd number of detector size can have problems for the moment.")
    end
    
    #------------- step1 padding
    # we assume that detcount is a power of 2
    # pad 2 times
    sz_pad = Int(2^ceil(log(detcount+1)/log(2)))
    sz_half = sz_pad >> 1 # even number
    padding = sz_pad - detcount
    padding_half = padding >> 1

    UU = zeros(nangles, sz_half)
    VV = zeros(nangles, sz_half)
    
    tau = cat(Array(1:2:Int(sz_pad/2-1)), Array(Int(sz_pad/2-1):-2:0), dims=1)
    spatial_filter = zeros(sz_pad)
    spatial_filter[1] = 0.25
    spatial_filter[2:2:end] .= -1.0 ./ (pi * tau).^2
    ramlak = fftshift( fft(spatial_filter) )
    # ramlak = fftshift(fft(ifftshift(spatial_filter))) # even function

    UU = zeros(Int, sz_pad)
    VV = zeros(Int, sz_pad)

    sz_window = 9
    sz_window_half = div(sz_window-1, 2)
    kernel, kernel_spatial = make_freq_window(sz_pad, sz_window_half)
    # ws ./= sum(ws)
    kernel_idx = CartesianIndices(kernel)

    # assume that H=detcount, W=detcount
    H, W = detcount, detcount
    rec_img = zeros(H,W,nslices)

    # 1:sz_pad - div(detcount, 2) + 0.5 - 1.0
    offset = div(sz_pad, 2) + 0.5
    dtheta = angles[2]-angles[1]
    halfsz = Int(floor(sz_window/2))

    UU0 = collect(1:sz_pad) .- offset
    one_index = CartesianIndex(1, 1)
    
    Threads.@threads for slice=1:nslices
        println("slice no. $slice")
        # (optional) add the current slice and the next slice
        # if slice < nslices-1
        #     p_pad[:,1:detcount] .= view(p,:,slice,:) + view(p,:,slice+1,:)
        # else
        #     p_pad[:,1:detcount] .= view(p,:,slice,:)
        # end

        # for multithreading, give up saving memory allocations
        Q = zeros(ComplexF64, sz_pad, sz_pad)
        q = zeros(sz_pad, sz_pad)
        p_pad = zeros(nangles, sz_pad)
        # fill!(Q, 0.0)
        # fill!(p_pad, 0.0)

        for i=1:detcount 
            p_pad[:, padding_half+1:end-padding_half] .= view(p,:,slice,:)
        end

        P = fftshift(fft(ifftshift(p_pad), 2)); 
        P .*= reshape(ramlak, (1,length(ramlak)))
        out_conv = zeros(ComplexF64, sz_pad+sz_window-1, sz_pad+sz_window-1)
            
        for iang=1:nangles
            cos_ = cos(angles[iang])
            sin_ = sin(angles[iang])
            
            #---------------------------
            # convolution of P[nangles x detcount] and C
            # continuous H = F * W (4)
            # discrete   Q = P * C (6)
            #---------------------------
            UU .= Int.(round.( UU0 .* cos_ .+ offset ) )
            # UU .-= sz_window_half
            UU[UU.>sz_pad] .-= sz_pad
            UU[UU.<1] .+= sz_pad

            VV .= Int.(round.( UU0 .* sin_ .+ offset ) )
            # VV .-= sz_window_half
            VV[VV.>sz_pad] .-= sz_pad 
            VV[VV.<1] .+= sz_pad
            
            P_idx = CartesianIndex.(VV,UU)
            fill!(out_conv, 0.0)
            
            for (idx_detcount, ci_P) in enumerate(P_idx)
                Pf = P[iang, idx_detcount]
                # println(ci_P)
                @simd for ci_C in kernel_idx
                     @inbounds out_conv[ci_P+ci_C-one_index] += Pf * kernel[ci_C] 
                end
            end
            
            Q .+= out_conv[halfsz+1:end-halfsz, halfsz+1:end-halfsz]
        end

        Q .*= dtheta / (detcount^2)
        q .= real( fftshift(ifft!(ifftshift(Q))) ./ kernel_spatial ) #./ kernel_weights )
        # q .*= size(Q, 2)
        # print(size(Q))
        # q[q .< 0] .= 0
        # q[q .> 1] .= 1

        rec_img[:,:,slice] .= reverse( q[padding_half+1:end-padding_half, padding_half+1:end-padding_half], dims=1 )        
    end

    return rec_img
end

"Gridrec for 2D image"
function recon2d_gridrec(p::Array{T, 2}, angles::Array{T, 1}) where {T<:AbstractFloat}
    p_3d_ = reshape(p, size(p)..., 1)
    p_3d = permutedims(p_3d_, 1, 3, 2)
    rec = recon2d_gridrec(p_3d, angles)
    return rec[:,:,1]
end
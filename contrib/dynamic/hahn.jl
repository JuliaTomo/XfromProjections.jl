using FFTW
using DSP
using ImageTransformations
using Dierckx
using Plots
using Colors
using PyCall
using LinearAlgebra
using CoordinateTransformations, Rotations, StaticArrays
using Dierckx

function rescale(x)
    mx = maximum(x)
    mi = minimum(x)
    return map(i-> (i-mi)/(mx-mi),x)
end

function ident(ϕ)
     t1(x) = x
     t2(x) = x
     return (t1,t2)
end

function nonlinear_transformation(ϕ)
    m1 = (sin(0.002*(ϕ*(257/(2*π))-1)*3/256))/4
    m2 = (sin(0.005*(ϕ*(257/(2*π))-1)*3/256))/4
    f(x) = x-128.5
    g1(x) = m1*x
    g2(x) = m2*x
    h(x) = x+1
    j(x) = x^5
    k(x) = x-1
    l1(x) = x/(5*m1)
    l2(x) = x/(5*m2)
    q(x) = x+128.5
    v1(x) = x |> f |> g1 |> h |> j |> k |> l1 |> q
    v2(x) = x |> f |> g2 |> h |> j |> k |> l2 |> q

    t(x) = SArray{Tuple{2},Float64,1,2}(v1(x[1]),v2(x[2]))
    return t
end

function nonlinear_transformation_inv(ϕ)
    m1 = (sin(0.002*(ϕ*(257/(2*π))-1)*3/256))/4
    m2 = (sin(0.005*(ϕ*(257/(2*π))-1)*3/256))/4
    f(x) = x+128.5
    g1(x) = x/m1
    g2(x) = x/m2
    h(x) = x-1
    j(x) = abs((x+0.0*im)^(1/5))*sign(x)
    k(x) = x+1
    l1(x) = x*(5*m1)
    l2(x) = x*(5*m2)
    q(x) = x-128.5
    v1(x) = x |> q |> l1 |> k |> j |> h |> g1 |> f
    v2(x) = x |> q |> l2 |> k |> j |> h |> g2 |> f

    return (v1,v2)
end

replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)

function precompute_kernels(ρ)
    f = fftfreq(length(ρ))
    fourier_filter = 2 * abs.(f)
    return fourier_filter
end

function filtering(data,kernel)
    FT = fft(data)
    g = real.(ifft(kernel.*FT))
    return g
end

function dynamic_backprojection()
end

function Hahn(sinogram, reconstrcution, inverse_motion)
    return reconstruction
end

function backprojection(data_filtered, θ, inv)
    (Γ_inv_x, Γ_inv_y) = inv(θ)
    output_size = length(data_filtered)
    mid_index = (output_size+1) / 2
    X = Γ_inv_x.(repeat(collect(1:output_size), 1, output_size))
    Y = Γ_inv_y.(repeat(collect(1:output_size)', output_size, 1))
    xpr = X .- mid_index
    ypr = Y .- mid_index
    t = ypr * cos(θ) - xpr * sin(θ)
    x = collect(1:output_size) .- mid_index
    spl = Spline1D(x, data_filtered[1,:], k=1, bc="zero")
    backprojected = spl.(t)
    return backprojected
end

function rod_phantom()
    rod = zeros(Float64, (600,600))
    rod[292:308,300:600] .= 1.0
    return rod
end

function disc_phantom(img, a, b, r)
    height, width = size(img)
    #find the possibility which has most surrounding pixels
    disc = zeros(Float64, (height,width))
    for x = 1:width
        for y = 1:height
            disc[y,x] = (x-a)^2 + (y-b)^2 < r^2 ? 1.0 : 0.0
        end
    end
    return disc
end

function FBP(sinogram, θ, ρ, inv)
    N = length(ρ)
    kernel = precompute_kernels(ρ)
    reconstruction = zeros(Float64,N,N)
    for i =1:length(θ)
        g = filtering(sinogram[:,i],kernel)
        reconstruction += backprojection(g', θ[i], inv)
    end
    return reconstruction * (π / (2 * length(θ)))
end

function radon(image,angles)
    theta = rad2deg.(angles)
    trans = pyimport("skimage.transform")
    sino = trans.radon(image, theta, true)
    return sino
end

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)

c_1 = 128.5
c_2 = 128.5
rad = 50

H,W = 256, 256

img = zeros(Float64, (H, W))
img = disc_phantom(img, c_1, c_2, rad)
p1 = plot(Gray.(img), aspect_ratio=:equal)
angles = collect(range(0,2*π, length=201))
pop!(angles)
bins = collect(-127.5:127.5)

sino = zeros(Float64,length(bins), length(angles))

anim = @animate for i = 1:length(angles)
    trans =nonlinear_transformation(angles[i])
    imgw = replace_nan(warp(img, trans, axes(img)))
    plot(Gray.(imgw), aspect_ratio=:equal, framestyle=:none)
    sino[:,i] = radon(imgw,[angles[i]])
end

gif(anim, "deformation.gif", fps = 20)

display(plot(Gray.(rescale(sino))))

reconstruction = FBP(sino, angles, bins, nonlinear_transformation_inv)
display(plot(Gray.(rescale(reconstruction))))

reconstruction_no_comp = FBP(sino, angles, bins, ident)
display(plot(Gray.(rescale(reconstruction_no_comp))))

using XfromProjections.curve_utils
using XfromProjections.snake_forward
using XfromProjections
using Logging
using IterTools
using LinearAlgebra
include("./sperm_phantom.jl")
include("./utils.jl")

cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "results/article_results/"))

#object to store center and radius of circle in complex plane
struct circle
    center::Complex{Float64}
    radius::Float64
end

#return the parametric equation for a circle in the complex plane
function get_parametric_circle_equation(c::circle)
    return t -> c.radius*ℯ^(t*im) + c.center
end

images, tracks = get_sperm_phantom(301,grid_size=0.1)

detmin, detmax = -38.0, 38.0
grid = collect(detmin:0.1:detmax)
bins = collect(detmin:0.125:detmax)
angles = collect(range(0.0,π,length=6))[1:end-1]
iterations = 1000
α = 0.01
β = 0.01
step_size = 0.2
r(s) = 1.0

num_points = 62
t = range(0, 2π, length=num_points)
c = circle((0.0-15.0*im), 15.0)
γ = get_parametric_circle_equation(c)
compl_circ = γ.(t)
outline_templ = cat(real(compl_circ), imag(compl_circ), dims=2)

frames2reconstruct = collect(1:10:300)
reconstructions = zeros(num_points,2,length(frames2reconstruct))
for (iter, frame_nr) in Base.Iterators.reverse(enumerate(frames2reconstruct))
    @info iter frame_nr

    #Get projection
    @info "making forward projection for frame: " frame_nr
    outline, normals = get_outline(tracks[frame_nr], r)
    sinogram = parallel_forward(outline, angles, bins)

    #Add noise
    @info "adding gaussian noise at level 0.01"
    rho = 0.01
    e = randn(size(sinogram));
    e = rho*norm(sinogram)*e/norm(e);
    sinogram = sinogram + e;

    @info "reconstructing"
    reconstruction = vedrana(outline_templ,angles,bins,sinogram, iterations, α, β, step_size)
    reconstructions[:,:,iter] = reconstruction

    @info "plotting"
    heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=true)
    plot!(reconstruction[:,1], reconstruction[:,2], aspect_ratio=:equal, linewidth=2, label=frame_nr)
    cd(savepath)
    savefig(@sprintf "result_original_vedrana_1%d" frame_nr)
end

global ps = AbstractPlot[]
for (iter, frame_nr) in enumerate(frames2reconstruct)
    best_recon = reconstructions[:,:,iter]

    l = @layout [a b c]
    p = heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    plot!(best_recon[:,1],best_recon[:,2], aspect_ratio=:equal, linewidth=5)
    push!(ps,p)
    if length(ps) == 3
        plot(ps[1], ps[2], ps[3], layout = l, size=(2000,600), linewidth=5)
        savefig(@sprintf "result_all_original_vedrana_1%d" frame_nr)
        global ps = AbstractPlot[]
    end
end
cd(cwd)

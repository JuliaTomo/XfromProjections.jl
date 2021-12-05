using Logging
using IterTools
using LinearAlgebra

include("../../phantoms/sperm_phantom.jl")
include("../snake_forward.jl")
include("../snake.jl")
include("../curve_utils.jl")

cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "result"))
!isdir(savepath) && mkdir(savepath)

#object to store center and radius of circle in complex plane
struct circle
    center::Complex{Float64}
    radius::Float64
end

#return the parametric equation for a circle in the complex plane
function get_parametric_circle_equation(c::circle)
    return t -> c.radius*ℯ^(t*im) + c.center
end


H,W = 609, 609
detmin, detmax = -38.0, 38.0
grid = collect(detmin:0.1:detmax)
r(s) = 1.0

images, tracks = get_sperm_phantom(301,r,grid)

grid = collect(detmin:0.1:detmax)
bins = collect(detmin:0.125:detmax)
angles = collect(range(0,π,length=9))[1:end-1]

@info  "using " length(angles) " angles"
iterations = 1000
α = 0.01
β = 0.3
step_size = 0.2
r(s) = 1.0
tol = 0.001

num_points = 62
t = range(0, 2π, length=num_points)
c = circle((0.0-0.0*im), 15.0)
γ = get_parametric_circle_equation(c)
compl_circ = γ.(t)
outline_templ = cat(real(compl_circ), imag(compl_circ), dims=2)

frames2reconstruct = collect(1:10:301)
reconstructions = zeros(num_points,2,length(frames2reconstruct))

best_residual = ones(length(frames2reconstruct),4)

for step_size in collect(1.0:-0.1:0.01)
    for α in collect(1.0:-0.1:0.01)
        for β in collect(1.0:-0.1:0.01)
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
                #plot(Gray.(sinogram./maximum(sinogram)))
                #cd(savepath)
                #savefig(@sprintf "sinogram_%d" frame_nr)

                @info "reconstructing"
                try
                    reconstruction, residuals = vedrana(outline_templ,angles,bins,sinogram, iterations, α, β, step_size, tol)


                if minimum(residuals) > 0.01 || minimum(residuals) > best_residual[iter,1]
                    @info "combination not better:" frame_nr step_size α β
                    continue
                else
                    best_residual[iter,:] =  [minimum(residuals), α, β, step_size]
                end

                reconstructions[:,:,iter] = reconstruction

                @info "plotting"
                heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=true)
                plot!(reconstruction[:,1], reconstruction[:,2], aspect_ratio=:equal, linewidth=2, label=frame_nr)
                plot!(outline_templ[:,1], outline_templ[:,2], aspect_ratio=:equal, linewidth=2, label="template")
                cd(savepath)
                savefig(@sprintf "result_original_vedrana_%d" frame_nr)

                @info "plotting"
                plot(collect(1:1:length(residuals)), residuals)
                cd(savepath)
                savefig(@sprintf "residuals_%d" frame_nr)
                catch
                    @info "combination not error:" frame_nr step_size α β
                    continue
                end
            end
        end
    end
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
        savefig(@sprintf "result_all_original_vedrana_%d_%d" length(angles) frame_nr)
        global ps = AbstractPlot[]
    end
end
cd(cwd)

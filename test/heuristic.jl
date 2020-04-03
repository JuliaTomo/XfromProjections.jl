using Logging
using Dierckx
using LinearAlgebra
using StaticArrays
using XfromProjections
using XfromProjections.curve_utils
using XfromProjections.snake_forward
using TomoForward
using ImageCore
using Suppressor
using Images
using LinearAlgebra
include("./sperm_phantom.jl")
include("./random_sperm.jl")
include("./utils.jl")

using Plots
using Colors
cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "results/article_results/"))


images, tracks = get_sperm_phantom(101,grid_size=0.1)

detmin, detmax = -38.0, 38.0
grid = collect(detmin:0.1:detmax)
bins = collect(detmin:0.125:detmax)

ang = π/2#π/5#2*π/3#π/4#π/2
angles, max_iter, stepsize = [ang], 10000, 0.1
tail_length = curve_lengths(tracks[end])[end]
num_points = 30
r(s) = 1.0
max_jiter = 500
frames2reconstruct = collect(1:10:100)
reconstructions = zeros(num_points,2,length(frames2reconstruct)+1)
#Add actual track at the end so rand sperm works and we can compare timewise
centerline_points = tracks[frames2reconstruct[end]+10]
t = curve_lengths(centerline_points)
spl = ParametricSpline(t,centerline_points',k=1, s=0.0)
tspl = range(0, t[end], length=num_points)
reconstructions[:,:,end] = spl(tspl)'
for (iter, frame_nr) in Base.Iterators.reverse(enumerate(frames2reconstruct))
    @info iter frame_nr
    jiter = 0

    #Get projection
    @info "making forward projection for frame: " frame_nr
    outline, normals = get_outline(tracks[frame_nr], r)
    projection = parallel_forward(outline, [ang], bins)

    #Add noise
    @info "adding gaussian noise at level 0.01"
    rho = 0.01
    e = randn(size(projection));
    e = rho*norm(projection)*e/norm(e);
    projection = projection + e;

    #Get ground truth of same length as recon
    t = curve_lengths(tracks[frame_nr])
    spl = ParametricSpline(t,tracks[frame_nr]', k=1, s=0.0)
    tspl = range(0, t[end], length=num_points)
    gt = spl(tspl)'

    angles = [ang]

    w_u = ones(num_points+2)
    w_u[1] = 0.0
    w_u[2] = 0.0
    w_l = ones(num_points+2)
    w_l[1] = 0.0
    w_l[2] = 0.0

    recon1_ok = false
    recon2_ok = false

    best_residual = Inf
    best_recon = get_straight_template(projection[:,1], r, [0.0 0.0], ang, num_points,bins)
    while jiter < max_jiter && best_residual > 1.5
        cd(savepath)
        jiter += 1
        @info jiter
        @info "setting up template"
        template = jiter==1 ? get_straight_template(projection[:,1], r, [0.0 0.0], ang, num_points,bins) : generate_random_sperm(projection[:,1], reconstructions[:,:,iter+1], ang, bins, r, num_points)

        @info "calculating initial reconstruction"
        #Reconstruct with weights only on one side
        recon1 = recon2d_tail(deepcopy(template),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, w_u, zeros(num_points+2))
        recon2 = recon2d_tail(deepcopy(template),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, zeros(num_points+2), w_l)
        best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)

        @info "checking if any parts could need mirroring 1"
        initial1 = deepcopy(recon1)
        initial2 = deepcopy(recon2)
        for flip_pt=1:num_points
            last_residual1, last_residual2 = residual1, residual2
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)
            #mirror and reconstruct with weights on both sides
            recon1 = recon2d_tail(deepcopy(recon1_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, zeros(num_points+2))
            recon2 = recon2d_tail(deepcopy(recon2_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, zeros(num_points+2), w_l)

            best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
        end

        # heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
        # plot!(recon1[:,1], recon1[:,2], aspect_ratio=:equal, label="recon1", linewidth=2)
        # plot!(recon2[:,1], recon2[:,2], aspect_ratio=:equal, label="recon2", linewidth=2)
        # plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, label="best", linewidth=2)
        # savefig(@sprintf "test%d" frame_nr)

        @info "keeping the best parts and restarting reconstruction 1"
        recon_best = keep_best_parts(residual1, deepcopy(best_recon), ang, bins, 3, num_points, tail_length, projection[:,1], r)
        recon1 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, w_u, zeros(num_points+2))
        recon2 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, zeros(num_points+2), w_l)
        best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)

        @info "checking if any parts could need mirroring 2"
        initial1 = deepcopy(recon1)
        initial2 = deepcopy(recon2)
        for flip_pt=1:num_points
            last_residual1, last_residual2 = residual1, residual2
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)
            #mirror and reconstruct with weights on both sides
            recon1 = recon2d_tail(deepcopy(recon1_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, zeros(num_points+2))
            recon2 = recon2d_tail(deepcopy(recon2_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, zeros(num_points+2), w_l)

            best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
        end
    end
    mirror = flip(best_recon,1,ang)
    next = reconstructions[:,:,iter+1]
    #recon = norm(best_recon-next) < norm(mirror-next) ? best_recon : mirror

    reconstructions[:,:,iter] = best_recon
    @info "plotting"
    #Plot the ground truth
    heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none)
    #plot the best reconstruction
    plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, label=best_residual, linewidth=2)
    plot!(mirror[:,1], mirror[:,2], aspect_ratio=:equal, label="mirror", linewidth=2)
    #plot!(next[:,1], next[:,2], aspect_ratio=:equal, label="next", linewidth=2)
    #plot!(mirror[:,1], mirror[:,2], aspect_ratio=:equal, label="mirror")
    #save the figure
    savefig(@sprintf "heuristic_500jiter_result_%d" frame_nr)
end

#global next = reconstructions[:,:,end]
# for (iter, frame_nr) in Base.Iterators.reverse(enumerate(frames2reconstruct))
#     best_recon = reconstructions[:,:,iter]
#     mirror = flip(best_recon,1,ang)
#     #recon = norm(best_recon-next) > norm(mirror-next) ? best_recon : mirror
#     heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
#     plot!(best_recon[:,1], best_recon[:,2], linewidth=2)
#     plot!(mirror[:,1], mirror[:,2], linewidth=2)
#     cd(savepath)
#     savefig(@sprintf "result_%d" frame_nr)
#     #global next = recon
# end
cd(cwd)

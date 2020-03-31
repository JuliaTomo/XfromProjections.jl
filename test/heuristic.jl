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

ang = 2*π/3
angles, max_iter, stepsize = [ang], 10000, 0.1
tail_length = curve_lengths(tracks[end])[end]
num_points = 30
r(s) = 1.0
max_jiter = 30
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
    best_recon = zeros(num_points,2)
    while jiter < max_jiter # && best_residual > 1.5

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

        @info "checking if any parts could need mirroring"
        initial1 = deepcopy(recon1)
        initial2 = deepcopy(recon2)
        for flip_pt=1:num_points
            last_residual1, last_residual2 = residual1, residual2
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)
            #mirror and reconstruct with weights on both sides
            recon1 = recon2d_tail(deepcopy(recon1_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)
            recon2 = recon2d_tail(deepcopy(recon2_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)

            best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
        end

        @info "keeping the best parts and restarting reconstruction"
        recon_best = keep_best_parts(residual1, deepcopy(best_recon), ang, bins, 3, num_points, tail_length, projection[:,1], r)
        recon1 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, w_u, zeros(num_points+2))
        recon2 = recon2d_tail(deepcopy(recon_best),r,[ang],bins,projection,max_iter, 0.0, stepsize, 1, zeros(num_points+2), w_l)
        best_residual,best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)

        @info "checking if any parts could need mirroring 2"
        initial1 = deepcopy(recon1)
        initial2 = deepcopy(recon2)
        for flip_pt=1:num_points
            last_residual1, last_residual2 = residual1, residual2
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)
            #mirror and reconstruct with weights on both sides
            recon1 = recon2d_tail(deepcopy(recon1_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)
            recon2 = recon2d_tail(deepcopy(recon2_flipped),r,[ang],bins,projection,100, 0.0, stepsize, 1, w_u, w_l)

            best_residual, best_recon[:,:], residual1, residual2 = try_improvement(best_residual, recon1, recon2, ang, bins, projection, best_recon, tail_length)
        end
    end
    mirror = flip(best_recon,1,ang)
    next = reconstructions[:,:,iter+1]
    #pick the one which is furthest closest to previous frame
    recon = norm(best_recon-next) < norm(mirror-next) ? best_recon : mirror

    reconstructions[:,:,frame_nr] = recon
    @info "plotting"
    #Plot the ground truth
    heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    #plot the best reconstruction
    plot!(recon[:,1], recon[:,2], aspect_ratio=:equal, label=best_residual)

    #plot!(mirror[:,1], mirror[:,2], aspect_ratio=:equal, label="mirror")
    #save the figure
    savefig(@sprintf "heuristic_result_%d" frame_nr)
end


cd(cwd)

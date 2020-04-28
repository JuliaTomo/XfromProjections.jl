module XfromProjections

# analytic
include("filter_proj.jl")
export filter_proj

# iterative
include("iterative/util_convexopt.jl")
include("iterative/tv_primaldual.jl")
include("iterative/sirt.jl")
export recon2d_tv_primaldual!, recon2d_sirt

# dynamic
include("dynamic/optical_flow.jl")
include("dynamic/tv_primaldual_flow.jl")
export recon2d_tv_primaldual_flow

# discrete
include("discrete/curve_utils.jl")
include("discrete/snake_forward.jl")
include("discrete/snake.jl")
export recon2d_tail, vedrana

#edges
include("edge_from_proj.jl")
export radon_log

end

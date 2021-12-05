# Welcome to XfromProjections.jl

XfromProjections provides image reconstruction from tomographic projection data. XfromProjections supports 2D image reconstructions for paralleal and fan beam and supports a stack of 2D images (3D images) slice by slice for paralleal beam. XfromProjections takes advantage of multi-threading. (To use multithreading, you can run julia with the option `julia -t2` e.g. if you want to use 2 threads.)

XfromProjectiions depends on [TomoForward](https://github.com/JuliaTomo/TomoForward.jl) package for forward operators of images.

## Install

Install [Julia](https://julialang.org/downloads/) and in [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/),

```
julia> ]
pkg> add https://github.com/JuliaTomo/TomoForward.jl
pkg> add https://github.com/JuliaTomo/XfromProjections.jl
```

## Examples and usages

Please see the codes in `examples` folder.

- `fbp.jl` : Filtered backprojection for 2D reconstruction
- `fbp_slices.jl` : Filtered backprojection for reconstructing a stack of 2D images
- `sirt2d.jl` : SIRT for 2D reconstruction
- `sirt2d_stack.jl` : SIRT for reconstructing a stack of 2D images
- `tv2d_primaldual.jl` : Total variation for 2D reconstruction
- `tv2d_stack_primaldual.jl` : Total variation for reconstructing a stack of 2D images
- `ctv2d_primaldual.jl` : L∞11 norm or total nuclear variation for spectral CT reconstruction

Regarding the code about the paper in submission "Material classification from sparse spectral X-ray CT using vectorial total variation based on L infinity norm", please refer to `ctv2d_primaldual.jl`.

# Features

## Image reconstruction from Projections

### Analytic methods

- FBP with different filters of Ram-Lak, Henning, Hann, Kaiser

### Iterative methods

- SIRT [Andersen, Kak 1984]
- Total Variation (TV) by primal dual solver [Chambolle, Pock 2011]
- Collaborative total variation (TNV) [Duran et al, 2016] (possibly for spectral CT)

## Shape form Projections

- (Todo) Parametric level set (Todo) []

## Contributions (please see `contrib` folders)

- Dynamic with optical flow constraint [Burger et al, 2017]


# Reference

- Andersen, A.H., Kak, A.C., 1984. Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm. Ultrasonic Imaging 6. https://doi.org/10.1016/0161-7346(84)90008-7
- Chambolle, A., Pock, T., 2011. A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging. Journal of Mathematical Imaging and Vision 40, 120–145. https://doi.org/10.1007/s10851-010-0251-1
- Duran, J., Moeller, M., Sbert, C., Cremers, D., 2016. Collaborative Total Variation: A General Framework for Vectorial TV Models. SIAM Journal on Imaging Sciences 9, 116–151. https://doi.org/10.1137/15M102873X
- Burger, M., Dirks, H., Frerking, L., Hauptmann, A., Helin, T., Siltanen, S., 2017. A variational reconstruction method for undersampled dynamic x-ray tomography based on physical motion models. Inverse Problems 33, 124008. https://doi.org/10.1088/1361-6420/aa99cf

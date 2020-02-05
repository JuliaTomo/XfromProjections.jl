Welcome to XfromProjections (under development)

XfromProjections aims to provide solutions X from tomographic projection data. X can be images, edges (or ). Instead of providing a solver as a blackbox, we sometimes divide the function into multiple steps. For example, for filtered back-projection, the user should use filtering and back projection seperately. 

XfromProjectiions depends on [TomoForward](https://github.com/JuliaTomo/TomoForward.jl) package for forward operators of images.

## Install

Install [Julia](https://julialang.org/downloads/) and in [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/),

```
julia> ]
pkg> add https://github.com/JuliaTomo/TomoForward.jl
pkg> add https://github.com/JuliaTomo/XfromProjections.jl
```

## Examples

Please see codes in test folder.


# Features

## Image reconstruction from Projections

### Analytic methods

- FBP with different filters of Ram-Lak, Henning, Hann, Kaiser [Kak]

### Iterative methods

- SIRT [3]
- TV using primal dual (Chambolle-Pock) method [4]

## Edges from projections

- Laplacian of Gaussian from projections [10]

## Shape form Projections

- (Todo) Parametric level set (Todo) []

# Todos

- 3D geometry
- Supporting GPU
- Forward projection of one closed mesh

# Reference

- [1] https://astra-toolbox.com
- [3] Andersen, A.H., Kak, A.C., 1984. Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm. Ultrasonic Imaging 6. https://doi.org/10.1016/0161-7346(84)90008-7
- [4] Chambolle, A., Pock, T., 2016. An introduction to continuous optimization for imaging. Acta Numerica 25, 161–319.
- [10] Srinivasa, N., Ramakrishnan, K.R., Rajgopal, K., 1992. Detection of edges from projections. IEEE Transactions on Medical Imaging 11, 76–80. https://doi.org/10.1109/42.126913

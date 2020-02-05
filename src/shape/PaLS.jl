"""
This paper aims to implement the method proposed in:

Aghasi, A., Kilmer, M., Miller, E.L., 2011. Parametric Level Set Methods for Inverse Problems. SIAM Journal on Imaging Sciences 4, 618–650. https://doi.org/10.1137/100800208

Kadu, A., van Leeuwen, T., Batenburg, K.J., 2018. A parametric level-set method for partially discrete tomography, in: DGCI.

Ref code:
https://github.com/ajinkyakadu125/ParametricLevelSet/blob/master/generateKernel.m
"""

Kernel(r) = max.(1.0-r, 0) .^8 .* (32*r.^3 .+ 25*r.^2 .+ 8*r .+ 1)

# switch rtype
#     case 'global'          % Global RBF (Gaussian)
#         kernelM = @(r) exp(-r.^2);
#         ki = 3.3;
#     case 'compact'          % Compactly-supported RBF (Wendland C4)
#         kernelM = @(r) max(1-r,0).^8.*(32*r.^3 + 25*r.^2 + 8*r + 1);
#         ki = 1;
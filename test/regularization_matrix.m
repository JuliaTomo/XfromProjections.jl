function [B,A] = regularization_matrix(N,alpha,beta)
%REGULARIZATION_MATRIX   Matrix for smoothing the snake.
%   B = REGULARIZATION_MATRIX(N,ALPHA,BETA)
%   B is an NxN matrix for imposing elasticity and rigidity to a snakes.
%   ALPHA is weigth for second derivative (elasticity)
%   BETA is weigth for (-)fourth derivative (rigidity)
%   Author: vand@dtu.dk

r = zeros(1,N);
r(1:3) = alpha*[-2 1 0] + beta*[-6 4 -1];
r(end-1:end) = alpha*[0 1] + beta*[-1 4];
A = toeplitz(r);
B = (eye(N)-A)^-1;

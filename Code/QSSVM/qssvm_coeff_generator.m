function [ G , y , S, tempG, tempS ] = qssvm_coeff_generator( X, Y )
%qssvm_coeff_generator 此处显示有关此函数的摘要
% X: n*m matrix, n training points in m dimensions ( must have been scaled to 0~1)
% Y: n*1 vecotr, n labels corresponding to X. (each element equals to -1 or 1)
% G: n*n matrix in the obj function
% y: 1*n vector of the constraints
% S: n*(m(m+3)/2) matrix of the constraints
% tempG: (m(m+3)/2)*(m(m+3)/2) matrix
% tempS: n*(m(m+3)/2 + m) matrix 
%
%   此处显示详细说明
% 1. first read my simplified note: 基于matlab的SVM设计 chapter2 & 3
% 2. Here we invoke the equation (3.10)
% 3. There are two output variables in this function: G y
% G here equals to 1/2*(S*inverse(tempG)*S') in equation (3.6)
% y here is the same as the y in equation (3.6)
% S will be used to decode the output of quadprog
% So does tempG Matrix

[tempG , tempS] = coeff_generator(X,Y);

[n] = size(X,1);

tempS = [tempS , X];
for i = 1:n
    S(i,:) = Y(i,1)*tempS(i,:);
end
G = 1/2*S*inv(tempG)*S';
G = (G + G')/2;
y = Y';



end


function [ Gnew , Snew] = Dagher_coeff_generator( X, Y )
%qssvm_coeff_generator 此处显示有关此函数的摘要
% X: n*m matrix, n training points in m dimensions ( must have been scaled to 0~1)
% Y: n*1 vecotr, n labels corresponding to X. (each element equals to -1 or 1)
% Gnew: (m(m+3)/2 + 1)*(m(m+3)/2 + 1) matrix in the obj function
% Snew: n*(m(m+3)/2 + m) matrix of the constraints
%
% 1. read my simplified note: 基于matlab的SVM设计 chapter2

[n,m] = size(X);

[tempG , tempS] = coeff_generator(X,Y);

Gnew = [tempG zeros(m*(m+3)/2,1);zeros(1,m*(m+3)/2) 0];

tempS = [tempS , X];
for i = 1:n
    tempS(i,:) = Y(i,1)*tempS(i,:);
end
Snew = [tempS, Y];

end


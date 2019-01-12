function [ A, b, c1 ] = Dagher_train( X , Y)
%qssvm_train: Given training points£º X, it produces each hypersplane
%with paramters: A, b, & c1. 

%%input
% X: normalized train dataset
% Y: its corresponding labels |  1 or -1

%%output
% A: m*m matrix in the obj function
% b: m*1 vector in the obj(b here equals to b' in the obj)
% c1: cosntant coeff in the obj

    [n,m] = size(X);
    [ Gnew, Snew] = Dagher_coeff_generator( X, Y );
    opts = optimset('Algorithm','interior-point-convex','Display','off');%interior-point-convex
    [znew, fval, eflag, output, lambda] = quadprog(Gnew,[],-Snew,-ones(n,1),[],[],[],[],[],opts);
    z = znew(1:end-1,1);
    c1 = znew(end,1);
    
    A = Uptriangle(z(1:m*(m+1)/2,1)');
    A = A + triu(A,1)';
    b = z(end-m+1:end);
end


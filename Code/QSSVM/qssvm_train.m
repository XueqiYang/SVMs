function [ A, b, c1 ] = qssvm_train( X , Y , c)
%qssvm_train: Given training points： X, it produces each hypersplane
%with paramters: A, b, & c1. 

%%input
% X: normalized train dataset
% Y: its corresponding labels |  1 or -1
% c: penalty parameter | equals to 2^logC

%%output
% A: m*m matrix in the obj function
% b: m*1 vector in the obj(b here equals to b' in the obj)
% c1: cosntant coeff in the obj

%   此处显示详细说明
    [n,m] = size(X);
    [ G , y ,S , tempG, tempS] = qssvm_coeff_generator( X, Y );
    opts = optimset('Algorithm','interior-point-convex','Display','off');%interior-point-convex
    [alpha, fval, eflag, output, lambda] = quadprog(G,-ones(n,1),[],[],y,0,zeros(n,1),c*ones(n,1),[],opts);
    z = inv(tempG)*S'*alpha/2;
    descend_alpha = sort(alpha,'descend');
   
    %index_corp = find((descend_alpha < c*0.9)&(descend_alpha > c*0.01));    
    index_corp = find(descend_alpha < c*0.9);
    if isempty(index_corp)
        index_corp = 1;
        fprintf('This train process may be unstable. And the penalty efficient: c = %d\n',c);
    end
    alpha_corp = descend_alpha(index_corp);
    
    index_sv = find(alpha == alpha_corp(1));
    c11 = Y(index_sv)- tempS(index_sv,:)*z;
    
    %decoding
    A = Uptriangle(z(1:m*(m+1)/2,1)');
    A = A + triu(A,1)';
    b = z(end-m+1:end);
    c1 = c11(1);
end


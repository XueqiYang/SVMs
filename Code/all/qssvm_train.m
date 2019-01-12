function [ A, b, c1 ] = qssvm_train( X , Y , c)
%QSSVM 此处显示有关此函数的摘要
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
        fprintf('c = %d\n',c);
    end
    alpha_corp = descend_alpha(index_corp);
    % here we cannot determine value 10 there lies a serious bug!!!
%    index_sv = zeros(1,5);
%     for i =1:5
%         index_sv(i) = find(alpha == descend_alpha(i));
%     end
    
    index_sv = find(alpha == alpha_corp(1));
    c11 = Y(index_sv)- tempS(index_sv,:)*z;
    
    %decoding
    A = Uptriangle(z(1:m*(m+1)/2,1)');
    A = A + triu(A,1)';
    b = z(end-m+1:end);
    c1 = c11(1);
end


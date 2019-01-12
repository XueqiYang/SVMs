function [ A, b, c1 ] = Dagher_train( X , Y)
%QSSVM �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
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


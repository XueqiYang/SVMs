function [ X_noised ] = addnoise( X , a )
%ADDNOISE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    [n,m] = size(X);
    for i = 1:m
        temp = unique(X(:,i));
        if(length(temp)<5)
        	X(:,i) = X(:,i) + 1/(length(temp)-1) * a * rand(n,1);
        end
    end
    X_noised = X;
end


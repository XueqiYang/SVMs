function [ G , S ] = coeff_generator( X, Y )
%COEFF_GENERATOR 此处显示有关此函数的摘要
% X: n*m matrix, n training points in m dimensions ( must have been scaled to 0~1)
% Y: n*1 vecotr, n labels corresponding to X. (each element equals to -1 or 1)
% G: (m(m+3)/2)*(m(m+3)/2) matrix in the obj function
% S: n*(m(m+3)/2) matrix of the constraints
%   此处显示详细说明
% 1. first read my simplified note: 基于matlab的SVM设计 chapter2 & 3
% 2. Here we invoke the equation (3.6)
% 3. But there is a subtle differece
% 4. There are two output variables in this function: G S
% G here is the same as the G in equation (3.6)
% but S here = [(s1)';(s2)';...;(sn)'];
% 
% You may need to generate what you want based on these two output vars.
% Do it by yourself!

    [n,m] = size(X);
    
    G = zeros(m*(m+3)/2,m*(m+3)/2);
    H = zeros(m,(m^2+3*m)/2,n);
    M = zeros(m,(m^2+m)/2,n);
    for q = 1:n
        for i = 1:m
            for j = 1:m
                if(i<j)
                    index_i = i;
                    index_j = j;
                else
                    index_i = j;
                    index_j = i;
                end
                p = (index_i-1)*(m+m-index_i+2)/2+(index_j-index_i+1);
                M(i,p,q) = X(q,j);
            end
        end
    end

    for q = 1:n
        H(:,:,q) = [M(:,:,q), eye(m)];
    end

    for q = 1:n
        G = G + H(:,:,q)'*H(:,:,q);
    end


    S = zeros(n,(m+1)*m/2);

    for q = 1:n
        cnt = 1;
        for i= 1:m
            for j = i:m
                if(i == j)
                    S(q,cnt) = X(q,i)*X(q,j)/2;
                else
                    S(q,cnt) = X(q,i)*X(q,j);
                end
                cnt = cnt + 1;
            end
        end
    end

end


function [AA] = Uptriangle(A)
    m=length(A);
    n=(-1+sqrt(1+8*m))/2 ; % 计算向量对应的上三角矩阵的维数
    AA=zeros(n,n);
    % 以下把向量转化为上三角
    for i=1:n
        for j=i:n
            index=sum(n:-1:n-i+2)+j-i+1;  
            AA(i,j)=A(index);       
        end
    end
end
function [AA] = Uptriangle(A)
    m=length(A);
    n=(-1+sqrt(1+8*m))/2 ; % ����������Ӧ�������Ǿ����ά��
    AA=zeros(n,n);
    % ���°�����ת��Ϊ������
    for i=1:n
        for j=i:n
            index=sum(n:-1:n-i+2)+j-i+1;  
            AA(i,j)=A(index);       
        end
    end
end
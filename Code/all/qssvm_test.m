function [ Sens, Spec, TestAcc ,BAC, Accuracy ] = qssvm_test( X, Y, W, b, c  )
%qssvm_coeff_generator 此处显示有关此函数的摘要
% X: n*m matrix, n testing points in m dimensions ( must have been scaled to 0~1)
% Y: n*1 vecotr, n labels corresponding to X. (each element equals to -1 or 1)
% W: m*m matrix in the obj function
% b: m*1 vector in the obj(b here equals to b' in the obj)
% c: coeff in the obj
%   此处显示详细说明
% 1. first read PhD Thesis of Luo Jian (Chapter 3.2)
% 2. Here we invoke the equation (3)
% 3. Accuracy is 0~100 which depicts qssvm prediction ability with the
%   specified coeffs(W,b,c).

    n = size(X,1);
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    for i = 1:n
        temp = (1/2*X(i,:)*W*X(i,:)' + X(i,:)*b + c)*Y(i);
        if(temp > 0)
            if( Y(i) > 0)
                TP = TP + 1;
            else
                TN = TN + 1;
            end
        else
            if ( Y(i) > 0)
                FN = FN + 1;
            else
                FP = FP + 1;
            end
        end
    end
    Sens = TP/(TP+FN)*100;
    Spec = TN/(FP+TN)*100;
    TestAcc = TP/(TP+FP) + TN/(FN+TN);
    TestAcc =TestAcc * 100;
    BAC = (Spec+Sens)/2;
    Accuracy = (TP+TN)/n*100;
end
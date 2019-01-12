function [ MeanAccuracy, StdAccuracy ] = Dagher_crossvalind( subset_unbiased, sublabel_unbiased, K_cv )
%QSSVM_CROSSVALIND 此处显示有关此函数的摘要
% function
% make crossvalind regarding the input args with mean & std accuracy
% estimation

% args
% subset_unbiased: n1*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% K_cv: K_cv-crossvalind. for the details about K-CrossValind，pls google it
% C: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".
% 

indices = crossvalind('Kfold',sublabel_unbiased,K_cv);


VectorAccuracy = zeros(1,K_cv);
for i = 1:K_cv
    test = (indices == i);
    train = ~test;
    trainset = subset_unbiased(train,:);
    trainlabel = sublabel_unbiased(train,1);
    testset = subset_unbiased(test,:);
    testlabel = sublabel_unbiased(test,1);
    
    [ W, b, c ] = Dagher_train( trainset , trainlabel);
    VectorAccuracy(i)  = qssvm_test( testset, testlabel, W, b, c  );
end

MeanAccuracy = mean(VectorAccuracy,2);
StdAccuracy = std(VectorAccuracy,0,2);

end


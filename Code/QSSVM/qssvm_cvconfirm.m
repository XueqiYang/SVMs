function [ MeanAcc , StdAcc ] = qssvm_cvconfirm( subset_unbiased, sublabel_unbiased, K, times, logC )
%QSSVM_CVCONFIRM 此处显示有关此函数的摘要
% function
% repeat crossvalind for several times regarding the input args with mean & std accuracy
% estimation

% args
% subset_unbiased: n1*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% K: K_cv-crossvalind. for the details about K-CrossValind，pls google it
% times: for repeating
% logC: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".
% MeanAcc: mean of accuracies
% StdAcc: std of accuracies

    VectorMeanAcc = zeros(times,1);
    VectorStdAcc = zeros(times,1);
    tic
    
    C = 2^logC;
    
    for i = 1:times
    [ VectorMeanAcc(i), VectorStdAcc(i) ] = qssvm_crossvalind( subset_unbiased, sublabel_unbiased, K, C );
    i
    end
    MeanAcc = mean(VectorMeanAcc);
    StdAcc = mean(VectorStdAcc);
    
    disp(num2str(MeanAcc));
    disp(num2str(StdAcc));
end


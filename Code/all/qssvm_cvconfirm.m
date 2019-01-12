function [ MeanAcc , StdAcc ] = qssvm_cvconfirm( dataset, labelset, K, times, logC )
%QSSVM_CVCONFIRM 此处显示有关此函数的摘要
%   此处显示详细说明
    [set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);
    
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


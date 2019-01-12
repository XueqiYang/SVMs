function [ VectorAccuracy, VectorMeanAcc ] = qssvm_scan( dataset, labelset, logC, K )
%QSSVM_SCAN 此处显示有关此函数的摘要
%   此处显示详细说明

    if(nargin <3)
        logC = 1:12;
        K = 10;
    end

    C = 2.^logC;
    nC = length(logC);
    VectorAccuracy = zeros(1,nC);
    VectorMeanAcc = zeros(1,nC);
    VectorStdAcc = zeros(1,nC);


    [set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);

    for i = 1:nC
        [ W, b, c ] = qssvm_train( subset_unbiased , sublabel_unbiased , C(i));
        [ ~, ~, ~ ,~, VectorAccuracy(i) ] = qssvm_test( subset_unbiased, sublabel_unbiased, W, b, c  );
        [ VectorMeanAcc(i), VectorStdAcc(i) ] = qssvm_crossvalind( subset_unbiased, sublabel_unbiased, K, C(i) );
        i
    end 
    
    figure;
    subplot(211)
    plot(logC,[VectorAccuracy;VectorMeanAcc]);
    subplot(212)
    plot(logC,VectorStdAcc);

end


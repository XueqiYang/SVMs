function [ VectorAccuracy, VectorMeanAcc ] = ClusterSVMQP_scan( subset_unbiased, sublabel_unbiased, clusterlabelset, K_cv, k, logC )
%QSSVM_SCAN 此处显示有关此函数的摘要
%   此处显示详细说明

    C = 2.^logC;
    nC = length(logC);
    VectorAccuracy = zeros(1,nC);
    VectorMeanAcc = zeros(1,nC);
    VectorStdAcc = zeros(1,nC);

    for i = 1:nC
        [ w ,b ] = ClusterSVMQP_train( subset_unbiased, sublabel_unbiased, clusterlabelset, k, C(i) );
        [ VectorAccuracy(i) ] = ClusterSVMQP_test( subset_unbiased, sublabel_unbiased, clusterlabelset, w, b );
        [ VectorMeanAcc(i), VectorStdAcc(i),~ ] = ClusterSVMQP_crossvalind( subset_unbiased, sublabel_unbiased, clusterlabelset, K_cv, k, C(i) );
        i
    end 
    
    figure;
    subplot(211)
    plot(logC,[VectorAccuracy;VectorMeanAcc]);
    subplot(212)
    plot(logC,VectorStdAcc);

end


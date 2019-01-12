function [ MeanAcc , StdAcc, MaxAcc, MinAcc, MeanTime ] = ClusterSVMQP_cvconfirm( subset_unbiased, sublabel_unbiased, clusterlabelset, K, k, times, logC )
%QSSVM_CVCONFIRM 此处显示有关此函数的摘要
%   此处显示详细说明    
    VectorMeanAcc = zeros(times,1);
    VectorTime = zeros(times,1);
    C = 2^logC;
    
    for i = 1:times
        [ VectorMeanAcc(i), VectorTime(i) ] = ClusterSVMQP_crossvalind( subset_unbiased, sublabel_unbiased, clusterlabelset, K, k, C );
        i
    end
    MeanAcc = mean(VectorMeanAcc);
    StdAcc = std(VectorMeanAcc);
    MaxAcc = max(VectorMeanAcc);
    MinAcc = min(VectorMeanAcc);
    MeanTime = mean(VectorTime);
    
    
    disp(['MeanAcc',num2str(MeanAcc)]);
    disp(['StdAcc',num2str(StdAcc)]);
    disp(['MaxAcc',num2str(MaxAcc)]);
    disp(['MinAcc',num2str(MinAcc)]);
    disp(['MeanTime',num2str(MeanTime)]);
end


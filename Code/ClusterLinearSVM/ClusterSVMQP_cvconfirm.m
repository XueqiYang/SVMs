function [ MeanAcc , StdAcc, MaxAcc, MinAcc, MeanTime ] = ClusterSVMQP_cvconfirm( subset_unbiased, sublabel_unbiased, clusterlabelset, K, k, times, logC )
%ClusterSVMQP_cvconfirm 
%%function
% invoking ClusterSVMQP_crossvalind repeatly.
% This function is to evaluate CSVM using normal 10-Crossvalind after settling
% down the optimized penalty efficient logC.

%%input
% subset_unbiased: n*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% clusterlabelset: indicates which label the points belong to
% K: K-fold-crossvalind. for the details about K-Fold-CrossValind£¬pls google it
% k: num of clusters
% times: for repeating
% logC: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".

%%ouput
% MeanAcc: mean of accuracies
% StdAcc: std of accuracies
% MaxAcc: max of accuracies
% MinAcc: min of accuracies
% MeanTime: mean of test time

    VectorMeanAcc = zeros(times,1);
    VectorTime = zeros(times,1);
    
    for i = 1:times
        [ VectorMeanAcc(i), VectorTime(i) ] = ClusterSVMQP_crossvalind( subset_unbiased, sublabel_unbiased, clusterlabelset, K, k, logC );
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


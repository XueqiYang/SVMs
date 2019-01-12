function [ MeanAccuracy, MeanTime ] = ClusterSVMQP_crossvalind( subset_unbiased, sublabel_unbiased, clusterlabelset, K_cv, k, logC )
%ClusterSVMQP_crossvalind 此处显示有关此函数的摘要
% function
% make crossvalind regarding the input args with mean & std accuracy
% estimation

% args
%%input
% subset_unbiased: n*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% clusterlabelset: indicates which label the points belong to
% k: num of clusters
% K_cv: K_cv-crossvalind. for the details about K-CrossValind，pls google it
% logC: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".

%%ouput
% MeanAccuracy: mean of accuracies
% MeanTime: mean of test time

indices = crossvalind('Kfold',sublabel_unbiased,K_cv);


VectorAccuracy = zeros(1,K_cv);
times = zeros(1,K_cv);
for i = 1:K_cv
    test = (indices == i);
    train = ~test;
    trainset = subset_unbiased(train,:);
    trainlabel = sublabel_unbiased(train,1);
    trainclusterlabel = clusterlabelset(train,1);
    testset = subset_unbiased(test,:);
    testlabel = sublabel_unbiased(test,1);
    testclusterlabel = clusterlabelset(test,1);
    
    [ w ,b ] = ClusterSVMQP_train( trainset, trainlabel, trainclusterlabel, k, logC );
    tic;
    [ VectorAccuracy(i) ] = ClusterSVMQP_test( testset, testlabel, testclusterlabel, w, b );
    times(i) = toc;
end

MeanAccuracy = mean(VectorAccuracy,2);
MeanTime = mean(times);
end


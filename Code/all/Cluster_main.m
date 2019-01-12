clear all; clc;
load '..\Dataset\german.mat'
labelset = german(:,1);
dataset = german(:,2:end);
clear german

% clear all; clc;
% load '..\Dataset\Australia.mat'
% labelset = Australia(:,1);
% dataset = Australia(:,2:end);
% clear Australia

[ set_biased, subset_unbiased, sublabel_unbiased ] = dataprep( dataset, labelset );

k = 2;

rng(1);
opts = statset('Display','final');
[cluterlabelset,centro] = kmeans(set_biased,k,'Replicates',5,'Options',opts);


%[ w ,b ] = ClusterQSSVM_train( set_biased, labelset, cluterlabelset, k, 4096 );
%[ Accuracy ] = CluterQSSVM_test( set_biased, labelset, cluterlabelset, w, b );

[ MeanAccuracy, StdAccuracy ] = ClusterSVMQP_crossvalind( set_biased, labelset, cluterlabelset, 10, 2, 2^11 )

[ VectorAccuracy, VectorMeanAcc ] = ClusterSVMQP_scan( set_biased, labelset, cluterlabelset, 10, 2, 4:2:20 )

[ MeanAcc , StdAcc, MaxAcc, MinAcc, MeanTime ] = ClusterSVMQP_cvconfirm( set_biased, labelset, cluterlabelset, 10, k, 10, 4 );

[ mAcc, stdAcc, maxAcc, minAcc, mTime ] = ClusterSVMQP_crossvalind_lj( set_biased, labelset, cluterlabelset, 10, k, 10,4:2:20 );


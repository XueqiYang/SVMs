clear all; clc;
load '..\Dataset\German.mat'
labelset = German(:,1);
dataset = German(:,2:end);
clear German

% clear all; clc;
% load '..\Dataset\Australia.mat'
% labelset = Australia(:,1);
% dataset = Australia(:,2:end);
% clear Australia

[ set_biased, subset_unbiased, sublabel_unbiased ] = dataprep( dataset, labelset );

k = 2;

%avoiding stochastic phenomenon
rng(1);
opts = statset('Display','final');
[cluterlabelset,centro] = kmeans(set_biased,k,'Replicates',5,'Options',opts);

%%find the best efficient: logC
[ VectorAccuracy, VectorMeanAcc ] = ClusterSVMQP_scan( set_biased, labelset, cluterlabelset, 10, 2, 4:2:20 );

%%Predict the best accuracy using normal 10-fold crossvalind for 10 times
[ MeanAcc , StdAcc, MaxAcc, MinAcc, MeanTime ] = ClusterSVMQP_cvconfirm( set_biased, labelset, cluterlabelset, 10, k, 10, 4 );

%%Predict the best accuracy using the updated 10-fold crossvalind for 10 times
[ mAcc, stdAcc, maxAcc, minAcc, mTime ] = ClusterSVMQP_crossvalind_updated( set_biased, labelset, cluterlabelset, 10, k, 10, 4:2:20 );

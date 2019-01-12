% 
% clear all;clc;
% load '..\Dataset\German.mat'
% 
% %%separate labelset & dataset
% labelset = German(:,1);
% dataset = German(:,2:end);
% clear German


clear all;clc;
load '..\Dataset\Australia.mat'

%%separate labelset & dataset
labelset = Australia(:,1);
dataset = Australia(:,2:end);
clear Australia

%%normalize dataset
[ set_biased, subset_unbiased, sublabel_unbiased ] = dataprep( dataset, labelset );

%%Updated crossvalind( sublte)

%%The input arguement: dataset_scale uses the dataprep function: SVMScale in this folder.
%[ mAcc, stdAcc, mtime ] = CrossValind_updated( dataset_scale, labelset, 10, -5:5, -5:5, 10);

%%The input arguement: set_biased uses the dataprep function: ..\Dataprep
[ mAcc, stdAcc, mtime ] = CrossValind_updated( set_biased, labelset, 10, -5:5, -5:5, 10);

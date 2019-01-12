% close all;
% clear all; clc;
% load '..\Dataset\bcw_dataset.mat'
% labelset = bcw_dataset(:,1);
% dataset = bcw_dataset(:,2:end);
% clear bcw_dataset
% % 
% clear all; clc;
% load '..\Dataset\iris.mat'
% labelset = iris(:,1);
% dataset = iris(:,2:end);
% clear iris.mat
% % 
% clear all; clc;
% load '..\Dataset\car_dmmy.mat'
% labelset = car_dmmy(:,1);
% dataset = car_dmmy(:,2:end);
% clear car_dmmy.mat

% clear all; clc;
% load '..\Dataset\germantestS2.mat'
% labelset = germantestS2(:,1);
% dataset = germantestS2(:,2:end);
% clear germantestS2

% clear all; clc;
% load german.mat
% labelset = german(:,1);
% dataset = german(:,2:end);
% clear german

% clear all; clc;
% load Australia.mat
% labelset = Australia(:,1);
% dataset = Australia(:,2:end);
% clear Australia

[set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);



%%this may takes a few minutes.
%%Randomly split set_biased into train & test sets with the proportion
%%efficient: k = trainset/(trainset + testset)
% for k = [0.1 0.2 0.4]
%     [meanAcc, stdAcc] = Dagher_confirm(set_biased, labelset, 100, k,'German');
% end


%%Repeat normal 10-fold crossvalind for 10 times to calculate the accuracy
%%vector
% Dagher_cvconfirm(set_biased, labelset, 10, 10);

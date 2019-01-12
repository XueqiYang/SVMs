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

clear all; clc;
load '..\Dataset\german.mat'
labelset = german(:,1);
dataset = german(:,2:end);
clear german

clear all; clc;
load '..\Dataset\Australia.mat'
labelset = Australia(:,1);
dataset = Australia(:,2:end);
clear Australia

[set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);


%%this may takes a few minutes.

%%find the best efficient logC
% qssvm_scan(set_biased, labelset, 2.5:1:10 ,10);

%%randomly split set_biased into train and test sets for 100 times with
%%logC equals to 4 and k wandering from 0.1 , 0.2 to 0.4.
% for k = [0.1, 0.2, 0.4]
%     [meanAcc, stdAcc] = qssvm_confirm(set_biased, labelset, 4, 100, k,'German');
% end

%%Normal 10-fold crossvalind repeated for 100 times with penalty efficient logC
%%= 4
% qssvm_cvconfirm(set_biased, labelset, 10, 100, 4);




%%this may takes a lot of minutes.
%%Updated 10-fold crossvalind for qssvm. It is repeated for 10 times.
[ MeanAccuracy, StdAccuracy ] = qssvm_crossvalind_updated(set_biased, labelset, 10, -5:15, 10);

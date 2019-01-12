% close all;
% clear all; clc;
% load bcw_dataset.mat
% labelset = bcw_dataset(:,1);
% dataset = bcw_dataset(:,2:end);
% clear bcw_dataset
% % 
% clear all; clc;
% load iris.mat
% labelset = iris(:,1);
% dataset = iris(:,2:end);
% clear iris.mat
% % 
% clear all; clc;
% load car_dmmy.mat
% labelset = car_dmmy(:,1);
% dataset = car_dmmy(:,2:end);
% clear car_dmmy.mat


% % 
% 
% 
% Dagher_cvconfirm(dataset, labelset, 10, 10);
% qssvm_cvconfirm(dataset, labelset, 10, 100, 4);

% % this may takes a few minutes.
% for k = [0.1 0.2 0.4]
%     [meanAcc, stdAcc] = Dagher_confirm(dataset, labelset, 100, k,'German');
% end
% qssvm_scan(dataset, labelset );
% qssvm_scan(dataset, labelset, 2.5:1:10 ,10);
% for k = [0.1, 0.2, 0.4]
%     [meanAcc, stdAcc] = qssvm_confirm(dataset, labelset, 4, 100, k,'German');
% end

% clear all; clc;
% load germantestS2.mat
% labelset = germantestS2(:,1);
% dataset = germantestS2(:,2:end);
% clear germantestS2

clear all; clc;
load german.mat
labelset = german(:,1);
dataset = german(:,2:end);
clear german

[set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);
%[ mAcc, stdAcc, mtime ] = CrossValind_lj1( set_biased, labelset, 10, -5:5, -5:5, 10);
disp(['Above is about German GaussKer '])


[ MeanAccuracy, StdAccuracy ] = qssvm_crossvalind_lj1(set_biased, labelset, 10, -5:15, 10);
disp(['Above is about German QSSVM '])

load Australia.mat
labelset = Australia(:,1);
dataset = Australia(:,2:end);
clear Australia
% 
% 
% % this may takes a few minutes.
[set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);
[ MeanAccuracy, StdAccuracy ] = qssvm_crossvalind_lj1(set_biased, labelset, 10, -5:15, 10);
disp(['Above is about Australia QSSVM '])

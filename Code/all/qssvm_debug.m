% load bcw_dataset.mat
% labelset = bcw_dataset(:,1);
% dataset = bcw_dataset(:,2:end);
% clear bcw_dataset
clear all;

X = [-1.5 -5;-2 -4;-3 -3;-4 -4;-5 -5;1 3;2 2;3 2;4 2;5 3];
Y = [1 1 1 1 1 -1 -1 -1 -1 -1]';

dataset = X;
labelset = Y;
C = 2560;

[set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);
[ W, b, c ] = qssvm_train( subset_unbiased , sublabel_unbiased , C);
VectorAccuracy  = qssvm_test( subset_unbiased, sublabel_unbiased, W, b, c  );
%[ MeanAccuracy, StdAccuracy ] = qssvm_crossvalind( subset_unbiased, sublabel_unbiased, 3, C );

figure;
plot(X(:,1)',X(:,2)','+')
hold on;

syms x1 y1 real;
% x1 =[-5:0.5:5];
% y1 = x1;
sm = [ x1; y1];
a_decision = 1/2*sm'*W*sm + b'*sm +c;
format long
ezplot(vpa(a_decision))


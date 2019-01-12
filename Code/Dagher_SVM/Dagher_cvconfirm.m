function [ MeanAcc , StdAcc ] = Dagher_cvconfirm( dataset, labelset, K, times )
%Dagher_cvconfirm �˴���ʾ�йش˺�����ժҪ
% function
% repeat K-fold crossvalind several times for evaluation

% args
% dataset: must be normalized
% labelset: corresponding labelset
% K: K_cv-crossvalind. for the details about K-CrossValind��pls google it
% times: for repeating

% MeanAcc: mean of Accuracy
% StdAcc�� std of accuracy

   set = dataset;
   label = labelset;
        
    VectorMeanAcc = zeros(times,1);
    VectorStdAcc = zeros(times,1);
    tic
    for i = 1:times
    [ VectorMeanAcc(i), VectorStdAcc(i) ] = Dagher_crossvalind( set, label, K);
    i
    end
    toc
    MeanAcc = mean(VectorMeanAcc);
    StdAcc = mean(VectorStdAcc);
    
    disp(num2str(MeanAcc));
    disp(num2str(StdAcc));
end


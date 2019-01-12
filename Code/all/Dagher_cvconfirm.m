function [ MeanAcc , StdAcc ] = Dagher_cvconfirm( dataset, labelset, K, times, cmd )
%QSSVM_CVCONFIRM 此处显示有关此函数的摘要
%   此处显示详细说明
    [set_biased, subset_unbiased, sublabel_unbiased] = dataprep(dataset,labelset);
    if cmd == 'biased'
        set = set_biased;
        label = labelset;
    else
        set = subset_unbiased;
        label = sublabel_unbiased;
    end
        
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


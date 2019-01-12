function [ set_biased, subset_unbiased, sublabel_unbiased ] = dataprep( dataset, labelset )
%DATAPREP 
% function:
% 1. Scale dataset to 0~1 to create set_biased.
% 2. use set_biased to create two balanced subsets: subset1 & subset2, that's the num of rows
% of subset1 equals to that of subset2
% 3. then combine subset1&2 together to create subset_unbiased for crossvalind
% methods
%
% arguments:
% dataset: n*m matrix
% labelset: n*1 vector. The corresponding label vector for dataset.(-1 or 1)
% set_biased: n*m matrix. num of Points in class1 may not equal that in class2
% subset_unbiased: t*m matrix. t<=n. A subset derived from set_biased.


    [n,m] = size(dataset);
    dataset = addnoise(dataset, 0.1);
    [dataset_scale,~] = mapminmax(dataset',0,1);
    set_biased = dataset_scale';
    
    dataset1 = set_biased((labelset == 1),:);
    dataset2 = set_biased((labelset ~= 1),:);
    n1 = size(dataset1,1);
    n2 = size(dataset2,1);
    if(n1>n2)
        minNum = n2;
    else
        minNum = n1;
    end
    subdataset1 = dataset1(1:minNum,:);
    subdataset2 = dataset2(1:minNum,:);

    subset_unbiased = [subdataset1;subdataset2];
    sublabel_unbiased = [ones(minNum,1);-1*ones(minNum,1)];
end


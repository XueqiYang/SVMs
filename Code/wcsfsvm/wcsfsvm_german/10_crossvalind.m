function [ MeanAccuracy, StdAccuracy ] = Dagher_crossvalind( subset_unbiased, sublabel_unbiased, K_cv )

indices = crossvalind('Kfold',sublabel_unbiased,K_cv);
%�����K_cv����Ӧ�õ���10���𣿺���ֲ�Ҫ�������µ���ƪ������������ˣ���Ϊ����Ҫ���������飬֮ǰ�Ĳ�����10��
VectorAccuracy = zeros(1,K_cv);

for i=1:K_cv
    test=(indices == i);
    train= ~test;
    trainset = subset_unbiased(train,:);
    trainlabel = sublabel_unbiased(train,1);
    testset = subset_unbiased(test,:);
    testlabel = sublabel_unbiased(test,1);
    

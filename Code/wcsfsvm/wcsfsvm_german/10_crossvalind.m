function [ MeanAccuracy, StdAccuracy ] = Dagher_crossvalind( subset_unbiased, sublabel_unbiased, K_cv )

indices = crossvalind('Kfold',sublabel_unbiased,K_cv);
%这里的K_cv不是应该等于10的吗？好奇怪不要按照最新的那篇论文里的来的了，因为我们要做对照试验，之前的不都是10吗
VectorAccuracy = zeros(1,K_cv);

for i=1:K_cv
    test=(indices == i);
    train= ~test;
    trainset = subset_unbiased(train,:);
    trainlabel = sublabel_unbiased(train,1);
    testset = subset_unbiased(test,:);
    testlabel = sublabel_unbiased(test,1);
    

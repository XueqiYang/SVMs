function [ MeanAccuracy, StdAccuracy ] = Dagher_confirm( dataset, labelset, times, k, fourline  )
%Dagher_confirm 此处显示有关此函数的摘要
% function
% evaluate dagher qsvm using random selection from dataset

% args
% dataset: n1*m matrix which comprises balanced two-class susbets
% labelset: corresponding labelset
% times: for repeating
% k: proportion = trainset/(trainset＋ testset)
% fourline: for display. Just use it to add comments

% MeanAccuracy: mean of Accuracy
% StdAccuracy： std of accuracy

    if(nargin <5)
        fourline = ' ';
    end

    VectorAccuracy = zeros(1,times);
    [set_biased, ~, ~] = dataprep(dataset,labelset);
    
    iCls1 = find(labelset==-1);
    iCls2 = find(labelset==1);
    nCls1 = length(iCls1);
    nCls2 = length(iCls2);
    minnCls = nCls1;
    if(nCls1 > nCls2)
        minnCls = nCls2;
    end
    nsubset = fix(minnCls * k);
    tic
    for i = 1:times
        rIndex1 = randperm(nCls1,nCls1);
        rIndex2 = randperm(nCls2,nCls2);
        train_setIndex = [iCls1(rIndex1(1:nsubset));iCls2(rIndex2(1:nsubset))];
        train_set = set_biased(train_setIndex,:);
        train_label = labelset(train_setIndex,:);

        test_setIndex = [iCls1(rIndex1(nsubset+1:end));iCls2(rIndex2(nsubset+1:end))];
        test_set = set_biased(test_setIndex,:);
        test_label = labelset(test_setIndex,:);
        [ W, b, c ] = Dagher_train( train_set , train_label);
        VectorAccuracy(i)  = qssvm_test( test_set, test_label, W, b, c  );
        fprintf('Completed: %.2f\n',i/times);
    end
    temp_toc = toc;
    disp('==================Result=======================') 
    disp(['平均运行时间',num2str(temp_toc/times),' s'])
    MeanAccuracy = mean(VectorAccuracy,2);
    StdAccuracy = std(VectorAccuracy,0,2);
    
    figure;
    plot(VectorAccuracy);
    xlabel('iteration');
    ylabel('accuracy');
    firstline = ['Repeat: ',num2str(times),'k:',num2str(k)];
    secondline = ['Avg:',num2str(MeanAccuracy),'%','Std:',num2str(StdAccuracy)];
    thirdline = ['平均运行时间',num2str(temp_toc/times),' s'];
    title({firstline;secondline;thirdline;['Dagher', fourline]});

end


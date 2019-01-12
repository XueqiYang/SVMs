function [ MeanAccuracy, StdAccuracy ] = qssvm_crossvalind_lj( subset_unbiased, sublabel_unbiased, K_cv, logC )
%QSSVM_CROSSVALIND 此处显示有关此函数的摘要
% function
% make crossvalind regarding the input args with mean & std accuracy
% estimation

% args
% subset_unbiased: n1*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% K_cv: K_cv-crossvalind. for the details about K-CrossValind，pls google it
% C: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".
% 

indices = crossvalind('Kfold',sublabel_unbiased,K_cv);


VectorAccuracy = zeros(1,K_cv);

for i = 1:K_cv
    i
    j = i + 1;
    if i == K_cv
        j = 1;
    end
    test = (indices == i);
    cv = (indices == j);
    train = ((indices ~= i) & (indices ~= j));
    
    trainset = subset_unbiased(train,:);
    trainlabel = sublabel_unbiased(train,1);
    testset = subset_unbiased(test,:);
    testlabel = sublabel_unbiased(test,1);
    cvset = subset_unbiased(cv,:);
    cvlabel = sublabel_unbiased(cv,1);
    
    bestAcc = 0;
    %using cvset to find out the best C for testset
    for C = 2.^logC
        [ W, b, c ] = qssvm_train( trainset , trainlabel , C);
        [ ~, ~, ~ , ~, Accuracy ] = qssvm_test( cvset, cvlabel, W, b, c  );
        if( bestAcc < Accuracy)
            bestAcc = Accuracy;
            bestC = C;
            bestW = W;
            bestb = b;
            bestc = c;
        end
    end
    bestAcc
    bestC
    
    tic;
    [ Sens(i), Spec(i), TestAcc(i), BAC(i), VectorAccuracy(i) ] = qssvm_test( testset, testlabel, bestW, bestb, bestc)
    times(i) = toc;
end

figure;
plot(VectorAccuracy);
MeanAccuracy = mean(VectorAccuracy,2)
StdAccuracy = std(VectorAccuracy,0,2)
MaxAccuracy = max(VectorAccuracy)
MinAcc = min(VectorAccuracy)
AverageTime = mean(times)
end


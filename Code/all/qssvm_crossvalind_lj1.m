function [ mAcc, stdAcc ] = qssvm_crossvalind_lj1( subset_unbiased, sublabel_unbiased, K_cv, logC, times )
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
MeanAccuracy = zeros(1,times);
AverageTime = zeros(1,times);

for q = 1:times

    indices = crossvalind('Kfold',sublabel_unbiased,K_cv);

    VectorAccuracy = zeros(1,K_cv);
    
    q
    for i = 1:K_cv
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

        tic;
        [ ~, ~, ~, ~, VectorAccuracy(i) ] = qssvm_test( testset, testlabel, bestW, bestb, bestc);
        times(i) = toc;
    end

    MeanAccuracy(q) = mean(VectorAccuracy,2);
    AverageTime(q) = mean(times);
end
    mAcc = mean(MeanAccuracy)
    maxAcc = max(MeanAccuracy)
    minAcc = min(MeanAccuracy)
    stdAcc = std(MeanAccuracy)
    mTime = mean(AverageTime)

end


function [ mAcc, stdAcc, mtime ] = CrossValind_updated( subset_unbiased, sublabel_unbiased, K_cv, logC, Gamma, times )
%QSSVM_CROSSVALIND 此处显示有关此函数的摘要
% function
% make crossvalind regarding the input args with mean & std accuracy
% estimation

% args
% subset_unbiased: n1*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% K_cv: K_cv-crossvalind. for the details about K-CrossValind，pls google it
% logC: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".
% Gamma: GausKer efficient
% times: for repeating

% mAcc: mean of Accuracies
% stdAcc: std of Accuracies
% mtime: mean of time



VectorAccuracy = zeros(1,K_cv);
MeanAccuracy = zeros(1,times);
MeanTime = zeros(1,times);
time_test = zeros(1, K_cv);
for q = 1:times
    q
    indices = crossvalind('Kfold',sublabel_unbiased,K_cv);

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
        for gamma = 2.^Gamma
            for C = 2.^logC
                cmd = ['-c ',num2str( C ),' -g ',num2str( gamma), '-q 1'];
                model = svmtrain(trainlabel,trainset,cmd);
                [predict_label, Accuracy, prob_estimates] = svmpredict(cvlabel,cvset,model,'-q 1');
                if( bestAcc < Accuracy(1))
                    bestAcc = Accuracy(1);
                    bestModel = model;
                    bestGamma = gamma;
                    bestC = C;
                end
            end
        end
        tic;
        [predict_label, Accuracy, prob_estimates ] = svmpredict(testlabel,testset,bestModel,'-q 1');
        time_test(i) = toc;
        VectorAccuracy(i) = Accuracy(1);

    end
    MeanTime(q) = mean(time_test);
    MeanAccuracy(q) = mean(VectorAccuracy,2);

end

mtime = mean(MeanTime)
mAcc = mean(MeanAccuracy)
maxAcc = max(MeanAccuracy)
minAcc = min(MeanAccuracy)
stdAcc = std(MeanAccuracy)
end

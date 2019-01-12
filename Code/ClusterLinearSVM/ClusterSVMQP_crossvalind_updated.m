function [ mAcc, stdAcc, maxAcc, minAcc, mTime ] = ClusterSVMQP_crossvalind_updated( subset_unbiased, sublabel_unbiased, clusterlabelset, K_cv, k, times,logC )
%ClusterSVMQP_crossvalind_updated
%%function
% using the updated crossvalind as suggested.
% 10 folds in total.
% 1 fold for cvset
% 1 fold for test set
% 8 fold for train set
% First we use train set and cvset to find out the best logC, then apply
% this efficent to get the hypersurface and consequently, the accuracy of
% the left test set can be obtained.
% It requires more minutes.

%%input
% subset_unbiased: n*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% clusterlabelset: indicates which label the points belong to
% k: num of clusters
% K_cv: K_cv-crossvalind. for the details about K-CrossValind£¬pls google it
% times: for repeating
% logC: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".

%%ouput
% mAcc: mean of accuracies
% stdAcc: std of accuracies
% maxAcc: max of accuracies
% minAcc: min of accuracies
% mTime: mean of test time


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
        trainclusterlabel = clusterlabelset(train,1);
        
        testset = subset_unbiased(test,:);
        testlabel = sublabel_unbiased(test,1);
        testclusterlabel = clusterlabelset(test,1);
        
        cvset = subset_unbiased(cv,:);
        cvlabel = sublabel_unbiased(cv,1);
        cvclusterlabel = clusterlabelset(cv,1);

        bestAcc = 0;
        %using cvset to find out the best C for testset
        for C = 2.^logC
            [ w ,b ] = ClusterSVMQP_train( trainset, trainlabel, trainclusterlabel, k, C );
            [ Accuracy ] = ClusterSVMQP_test( cvset, cvlabel, cvclusterlabel, w, b );
            if( bestAcc < Accuracy)
                bestAcc = Accuracy;
                bestw = w;
                bestb = b;
                bestC = C;
            end
        end

        tic;
        [ VectorAccuracy(i) ] = ClusterSVMQP_test( testset, testlabel, testclusterlabel, bestw, bestb );
        times(i) = toc;
        
        bestC
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


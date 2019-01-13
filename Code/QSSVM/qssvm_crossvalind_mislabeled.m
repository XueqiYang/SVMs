function [ mAcc, stdAcc ] = qssvm_crossvalind_mislabeled( subset_unbiased, sublabel_unbiased, K_cv, logC, times )
%qssvm_crossvalind_updated 
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


%%args
% subset_unbiased: n1*m matrix which comprises balanced two-class susbets
% sublabel_unbiased: corresponding labelsets
% K_cv: K_cv-crossvalind. for the details about K-CrossValind£¬pls google it
% logC: penalty coeff. for the details about penalty coeff, pls google "Soft SVM".
% times: for repeating
% mAcc: mean of accuracies
% stdAcc: std of accuracies

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
        
         % mislabel the data by Sherry
        len = length(trainlabel);  
        len_get = floor(len*0.1);     % round or cell will also works well as floor
        rand_index = randperm(len);   % sort the index randomly
        draw_rand_index = rand_index(1:len_get); % get the frontal 10% of training set
        for m = 1:len_get
            index1 = draw_rand_index(m);            
            trainlabel(index1) = -1*trainlabel(index1);
            %if trainlabel(m)==-1;
            %    trainlabel(m) = 1;
            %elseif trainlabel(m) == 1;
            %    trainlabel(m) = -1;
            %end
        end
        %  added by Sherry 
        
        % mislabel the data by Sherry, discarded
        len = length(trainlabel);  
        len_get = floor(len*0.1);     % round or cell will also works well as floor
        rand_index = randperm(len);   % sort the index randomly
        draw_rand_index = rand_index(1:len_get); % get the frontal 10% of training set
        for m = 1:draw_rand_index
            if trainlabel(m)==-1
                trainlabel(m)=1;
            elseif trainlabel(m)==1
                trainlabel(m)=-1;
            end
        end
        %  added end by Sherry, discarded        
        
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

 = min(MeanAccuracy)
    stdAcc = std(MeanAccuracy)
    mTime = mean(AverageTime)

end


function [ avg_predict_rate,std_predict_rate  ] = SVMAccPredict(labelset,dataset_scale,k,n,cmd )
%Make prediction of Targeted SVM by selecting trainset randomly and equally
%from two classes in dataset
%Detailed explanation goes here
%  default parameters:
%     k = [0.1,0.2,0.4];
%     n = 100;
%     cmd = ['-q 1'];-c 4 -g 0.25

    if nargin < 3
        k = [0.1,0.2,0.4];
        n = 100;
        cmd = ['-q 1'];
    end
    
    [mset,nset] = size(dataset_scale);


    iCls1 = find(labelset==-1)';
    iCls2 = find(labelset==1)';
    nCls1 = length(iCls1);
    nCls2 = length(iCls2);

    avg_predict_rate = zeros(1,length(k));
    std_predict_rate = zeros(1,length(k));
    for cnt = 1:length(k)
        predict_rate = zeros(1,n);
        ntrain = fix(k(cnt)*mset/2);
        for i = 1:n
            rIndex1 = randperm(nCls1,nCls1);
            rIndex2 = randperm(nCls2,nCls2);
            train_setIndex = [iCls1(rIndex1(1:ntrain)),iCls2(rIndex2(1:ntrain))];
            train_set = dataset_scale(train_setIndex,:);
            train_label = labelset(train_setIndex,:);

            test_setIndex = [iCls1(rIndex1(ntrain+1:end)),iCls2(rIndex2(ntrain+1:end))];
            test_set = dataset_scale(test_setIndex,:);
            test_label = labelset(test_setIndex,:);

            model = svmtrain(train_label,train_set,cmd);
            [predict_label, accuracy,prob_estimates] = svmpredict(test_label,test_set,model,'-q 1');
            predict_rate(1,i) = accuracy(1,1);
        end

        avg_predict_rate(1,cnt)  = mean(predict_rate');
        std_predict_rate(1,cnt) = std(predict_rate');
        figure;
        plot(predict_rate);
        xlabel('times');
        ylabel('predict_rate');
        firstline = ['Repeat: ',num2str(n)];
        secondline = ['k:',num2str(k(cnt))];
        thirdline = ['Average of predict rate:',num2str(avg_predict_rate(1,cnt)),'%'];
        fourthline = ['Std of predict rate:',num2str(std_predict_rate(1,cnt))];
        title({firstline;secondline;thirdline;fourthline});
    end
end


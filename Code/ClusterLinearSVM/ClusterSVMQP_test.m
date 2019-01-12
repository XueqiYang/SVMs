function [ Accuracy ] = ClusterSVMQP_test( testset_scale, testset_label, testCluster_label, w, b )
%ClusterSVMQP_test Given testset and its two labels and the correpsonding
%hyperplane efficients, the Accuracy is produced.

% testset_scale: normalized testset points: n by m matrix with n points, m dimensions.
% testset_label: 1 or -1, n by 1 column vector.
% testCluster_label: 1, 2,3 ...,numOfClusters, also a column vector.
% w: A matrix with numOfClusters-by-size(testset_scale,2)
% b: A column numOfClusters-by-1 vector.

% Accuracy: based on testset

[n,m] = size(testset_scale);
TP = 0;
TN = 0;
FP = 0;
FN = 0;
for i = 1:n
    j = testCluster_label(i);
    wj = w(j,:);
    bj = b(j,:);
    
    temp = (wj*testset_scale(i,:)' + bj)*testset_label(i);
    %predict successfully
    if(temp > 0)
        if( testset_label(i) > 0)
            TP = TP + 1;
        else
            TN = TN + 1;
        end
    else
        if ( testset_label(i) > 0)
            FN = FN + 1;
        else
            FP = FP + 1;
        end
    end
    
   
end
 Accuracy = (TP+TN)/n*100;

end


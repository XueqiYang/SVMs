function [ Accuracy ] = ClusterSVMQP_test( testset_scale, testset_label, testCluster_label, w, b )
%DATACLUSTER Given clustered training points, it produces each hypersplane
%with paramters: w & b. Each Row j of w and b represents the hyperslane of
%cluster j. j= 1 to k, with k defaults to 2.
% w: A matrix with k-by-size(testset_scale,2)
% b: A column k-by-1 vector.
% C: penalty parameter for qssvm



%opts = statset('Display','final');
%[clusterlabelset,centro] = kmeans(testset_scale,k,'Replicates',5,'Options',opts);

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


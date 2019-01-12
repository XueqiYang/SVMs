function [ w ,b ] = ClusterSVMQP_train( dataset_scale, labelset, clusterlabelset, k, logC )
%ClusterSVMQP_train: Given clustered training points, it produces each hypersplane
%with paramters: w & b. Each Row j of w and b represents the hyperslane of
%cluster j. j= 1 to k, with k defaults to 2.

%%input
% dataset_scale: normalized_dataset
% labelset: its corresponding labels |  1 or -1
% clusterlabelset: indicates which cluster the point belongs to
% k: numOfClusters
% logC: penalty parameter for qssvm

%%output
% w: A matrix with k-by-size(dataset_scale,2)
% b: A column k-by-1 vector.


C = 2.^logC;

[n,m] = size(dataset_scale);
rowH =repmat(-eye(m),1,k);
H = [(k+1)*eye(m), rowH; rowH',eye(k*m)];
H =  blkdiag(H, zeros(n+k,n+k));
f = [zeros(1,m) zeros(1,k*m) C*ones(1,n) zeros(1,k)]';
vector = zeros(n, m*(k+1)+n+k);

for q = 1:n
    j = clusterlabelset(q);
    vector(q,:) = [zeros(1,m), zeros(1,(j-1)*m), -labelset(q)*dataset_scale(q,:), ...
        zeros(1,(k-j)*m), zeros(1,q-1), -1, zeros(1,n-q), zeros(1,j-1), -labelset(q), zeros(1,k-j)];
end
lb = [ -inf(1,m), -inf(1,m*k), zeros(1,n), -inf(1,k) ]';

opts = optimset('Algorithm','interior-point-convex','Display','off');%interior-point-convex
[alpha, ~, ~, ~, ~] = quadprog(H, f, vector, -ones(n,1), [], [], lb, [], [],opts);

%w_c = alpha(1:m,1)';

w = ones(k,m);

for j = 1:k
    w(j,:) = alpha(j*m+1:j*m+m,1)';
end

%yita = alpha(k*m+m+1:k*m+m+n,1);
b = alpha(end-k+1:end,1);

end


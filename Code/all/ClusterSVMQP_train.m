function [ w ,b ] = ClusterSVMQP_train( dataset_scale, labelset, clusterlabelset, k, C )
%DATACLUSTER Given clustered training points, it produces each hypersplane
%with paramters: w & b. Each Row j of w and b represents the hyperslane of
%cluster j. j= 1 to k, with k defaults to 2.
% w: A matrix with k-by-size(dataset_scale,2)
% b: A column k-by-1 vector.
% C: penalty parameter for qssvm



%opts = statset('Display','final');
%[clusterlabelset,centro] = kmeans(dataset_scale,k,'Replicates',5,'Options',opts);

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


% figure;
% plot(dataset_scale(:,1),dataset_scale(:,2),'r.','MarkerSize',12)
% hold on;
% 
% syms x1 y1 real;
% % x1 =[-5:0.5:5];
% % y1 = x1;
% sm = [ x1; y1];
% for j = 1:k
%     a_decision(j) = w(j,:)*sm + b(j);
%     format long
%     ezplot(vpa(a_decision(j)))
% end
end


clear all;close all;clc;

X = [-1.5 -5;-2 -4;-3 -3;-4 -4;-5 -5;1 3;2 2;3 2;4 2;5 3];
Y = [1 1 -1 -1 -1 -1 -1 -1 1 1]';

figure;
scatter(X(:,1)',X(:,2)')
title('original data scatter')

figure
plot(X(Y==1,1),X(Y==1,2),'r.','MarkerSize',12)
hold on;
plot(X(Y==-1,1),X(Y==-1,2),'b.','MarkerSize',12)
title('true scatter')

[ set_biased, subset_unbiased, sublabel_unbiased ] = dataprep( X, Y );
%set_biased = X;

dataset = set_biased;
labelset = Y;

k = 2;

rng(1);
opts = statset('Display','final');
[idx,centro] = kmeans(dataset,k,'Replicates',5,'Options',opts);

figure;
plot(dataset(idx==1,1),dataset(idx==1,2),'r.','MarkerSize',12)
hold on
plot(dataset(idx==2,1),dataset(idx==2,2),'b.','MarkerSize',12)
plot(centro(:,1),centro(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title('Cluster Assignments and Centroids')
hold off



[n,m] = size(dataset);
C = 64;

rowH =repmat(-eye(m),1,k);
H = [(k+1)*eye(m), rowH; rowH',eye(k*m)];
H =  blkdiag(H, zeros(n+k,n+k));

f = [zeros(1,m) zeros(1,k*m) C*ones(1,n) zeros(1,k)]';
vector = zeros(n, m*(k+1)+n+k);

for q = 1:n
    j = idx(q);
    vector(q,:) = [zeros(1,m), zeros(1,(j-1)*m), -labelset(q)*dataset(q,:), ...
        zeros(1,(k-j)*m), zeros(1,q-1), -1, zeros(1,n-q), zeros(1,j-1), -labelset(q), zeros(1,k-j)];
end
lb = [ -inf(1,m), -inf(1,m*k), zeros(1,n), -inf(1,k) ];

opts = optimset('Algorithm','interior-point-convex');%interior-point-convex
[alpha, ~, ~, ~, ~] = quadprog(H, f, vector, -ones(n,1), [], [], lb, [], [],opts);

w_c = alpha(1:m,1)';

w = ones(k,m);

for j = 1:k
    w(j,:) = alpha(j*m+1:j*m+m,1)';
end

yita = alpha(k*m+m+1:k*m+m+n,1);
b = alpha(end-k+1:end,1);

%Now we draw the figure;
figure;
plot(dataset(:,1),dataset(:,2),'r.','MarkerSize',12)
hold on;

syms x1 y1 real;
% x1 =[-5:0.5:5];
% y1 = x1;
sm = [ x1; y1];
for j = 1:k
    a_decision(j) = w(j,:)*sm + b(j);
    format long
    ezplot(vpa(a_decision(j)))
end
function [ X_noised ] = addnoise( X , a )
%ADDNOISE We find that when using dummy variables, quadprog function
%cannnot be used with convex-point-algorithm. As dummy variables equal to 1
%or 0. We try to add noise to these variables. Due to the following split
%process, we cannot just consider dummy variables. Consider this situation:
%A variable equals to 0, 1 or 2. However, after set splitting, the variable
%in the training process equals to 0 or 1 other than 2. This will still
% leads quadprog into error. So we extend the condition into: length(temp)<5

    [n,m] = size(X);
    for i = 1:m
        temp = unique(X(:,i));
        if(length(temp)<5)
        	X(:,i) = X(:,i) + 1/(length(temp)-1) * a * rand(n,1);
        end
    end
    X_noised = X;
end


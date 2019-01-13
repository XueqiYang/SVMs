clear all; clc;
load('../Dataset/Australia.mat');
label = Australia(:, 1);
samples = Australia(:, 2:end);
clear Australia

[norm_samples,~] = mapminmax(samples',0,1);
samples = norm_samples';


N=length(label);
K=10;



t=cputime;

rng('default');
K_cv = 10;

% Do parameter tunning !
% parameters: 
% logC = 3.5;
% logbeta = 3.5;
% logp = -3;

logC = 1:1:10;
logbeta = -5:1:5;
logp = -5:1:5;

iC = 1:length(logC);
ibeta = 1:length(logbeta);
ip = 1:length(logp);

array_m = zeros(length(logC), length(logbeta), length(logp));

for iC= 1:length(logC)   %1;  C
    for ibeta = 1:length(logbeta)
	for ip  = 1:length(logp)
% beta= 2^-3.4; %-3.4 ;% 0.03; beta
% p = 2^-4;
	    C = 2 ^ logC(iC);
	    beta = 2 ^ logbeta(ibeta);
	    p = 2 ^ logp(ip);
	    
	    C2= C;
	    C1 = beta;
	    
	    % 
	    % accuracy
	    tem12=zeros(K,1);
	    
	    for kk=1:K
	    	indices = crossvalind('Kfold', label, K_cv);

	        % get the training data
	        tmp_xlabel = label(indices == 1);
	        xlabel = [ tmp_xlabel(tmp_xlabel == 1, :); tmp_xlabel(tmp_xlabel ~= 1, :)]; 
	    
	        xtrain = samples(indices == 1,:);
	        xtrain = [ xtrain(tmp_xlabel == 1, :); xtrain(tmp_xlabel ~= 1, :)];
	    
	        
	        % get the test data
	        ytest = samples(indices ~= 1, :);
	        ylabel = label(indices ~= 1);
	    
	        %  test the obtained classifier on the whole dataset
	        % u=samples;
	        % D1 = label;
	        u = ytest;
	        D1 = ylabel;
	    
	    
	        n1=size(xtrain,1);
	        n0=sum(xlabel == 1);
	    
	        Y = diag(xlabel);
	    
	        % svdd for getting the radius R0
	    
	        LB=zeros(n0,1);
	        UB=C*ones(n0,1);
	    
	        Aeq=ones(1,n0);
	        beq=1;
	    
	        f=zeros(n0,1);
	        for i=1:n0
	          f(i)=-gausskernel(xtrain(i,:),xtrain(i,:), p);
	        end
	    
	        H=zeros(n0);
	        for i=1:n0
	            for j=1:n0
	               H(i,j)=2*gausskernel(xtrain(i,:),xtrain(j,:), p);
	            end 
	        end
	    
	         opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
	         z1 = quadprog(H,f,[],[],Aeq,beq,LB,UB,[],opts);
	    
	         for i=1:n0
	            if z1(i)>=1e-3
	                temp=0;
	                for j=1:n0
	                temp=temp+z1(j)*gausskernel(xtrain(i,:),xtrain(j,:), p);
	                end
	               R0=sqrt(gausskernel(xtrain(i,:),xtrain(i,:), p)-2*temp+0.5*z1'*H*z1); 
	               break
	            end
	    
	    
	         end
	    
	        n2=n1-n0; 
	    
	        % svdd for getting the radius R2
	        LB=zeros(n2,1);
	        UB=C*ones(n2,1);
	    
	        Aeq=ones(1,n2);
	        beq=1;
	    
	        f=zeros(n2,1);
	        for i=1:n2
	          f(i)=-gausskernel(xtrain(n0+i,:),xtrain(n0+i,:), p);
	        end
	    
	        H=zeros(n2);
	        for i=1:n2
	            for j=1:n2
	               H(i,j)=2*gausskernel(xtrain(n0+i,:),xtrain(n0+j,:), p);
	            end 
	        end
	    
	         opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
	         z1 = quadprog(H,f,[],[],Aeq,beq,LB,UB,[],opts);
	    
	         for i=1:n2
	            if z1(i)>=1e-3
	                temp=0;
	                for j=1:n2
	                temp=temp+z1(j)*gausskernel(xtrain(n0+i,:),xtrain(n0+j,:), p);
	                end
	               R2=sqrt(gausskernel(xtrain(n0+i,:),xtrain(n0+i,:), p)-2*temp+0.5*z1'*H*z1); 
	               break
	            end
	    
	         end
	    
	         % calculate the distance for each point in the feature space
	    
	         d1=zeros(n0,1);
	         temp1=0;
	         for i=1:n0
	            for j=1:n0
	            temp1=temp1+gausskernel(xtrain(i,:),xtrain(j,:), p);
	            end
	         end
	    
	         for i=1:n0
	    
	             temp2=0;
	             for j=1:n0
	                 temp2=temp2+gausskernel(xtrain(i,:),xtrain(j,:), p);
	             end
	            d1(i)=sqrt(gausskernel(xtrain(i,:),xtrain(i,:), p)-2/n0*temp2+1/n0^2*temp1);
	    
	         end
	    
	          d2=zeros(n2,1);
	         temp1=0;
	         for i=1:n2
	            for j=1:n2
	            temp1=temp1+gausskernel(xtrain(i+n0,:),xtrain(j+n0,:), p);
	            end
	         end
	    
	         for i=1:n2
	             temp2=0;
	             for j=1:n2
	                 temp2=temp2+gausskernel(xtrain(i+n0,:),xtrain(j+n0,:), p);
	             end
	            d2(i)=sqrt(gausskernel(xtrain(i+n0,:),xtrain(i+n0,:), p)-2/n2*temp2+1/n2^2*temp1); 
	    
	         end
	    
	        % calculate the membership for each training point
	         s=zeros(n1,1);
	         for i=1:n0
	             if d1(i)<=R0
	                s(i)=0.6*(R0-d1(i))/(R0+d1(i))+0.4;
	             else
	                s(i)=0.4/(1+d1(i)-R0);
	             end  
	         end
	    
	         for i=n0+1:n1
	             if d2(i-n0)<=R2
	                s(i)=0.6*(R2-d2(i-n0))/(R2+d2(i-n0))+0.4;
	             else
	                s(i)=0.4/(1+d2(i-n0)-R2);
	             end  
	         end
	        % s=ones(n1,1);
	        % initialize the constant matrix for wcsfsvm model
	    
	        G=zeros(n1);
	        for i=1:n1
	            for j=1:n1
	                G(i,j)=gausskernel(xtrain(i,:),xtrain(j,:), p);
	            end
	        end
	    
	        I1=eye(n0);
	        L1=zeros(n0);
	        for i=1:n0
	            for j=1:n0
	             L1(i,j)=1/n0; 
	            end
	        end
	        K1=G(1:n1,1:n0);
	    
	        I2=eye(n2);
	        L2=zeros(n2);
	        for i=1:n2
	            for j=1:n2
	             L2(i,j)=1/n2; 
	            end
	        end
	         K2=G(1:n1,n0+1:n1);
	    
	         Q=Y*(G+C1*(n0/n1)*K1*(I1-L1)*K1'+C1*(n2/n1)*K2*(I2-L2)*K2')*Y;
	    
	        % use quadprog to solve the wcsfsvm model
	        A1=zeros(n1);
	        for i=1:n1
	           A1(i,:)=-Y(i,i)*(diag(Y).*G(:,i))';
	        end
	        A2=-diag(Y);
	        A3=-eye(n1);
	    
	        A=[A1,A2,A3];
	        b=-ones(n1,1);
	    
	        LB=[-Inf*ones(n1+1,1);zeros(n1,1)];
	    
	        f=zeros(2*n1+1,1);
	        f(n1+2:2*n1+1)=C2*s.*ones(n1,1);
	        H=zeros(2*n1+1);
	        H(1:n1,1:n1)=Q;
	    
	        opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
	        alpha = quadprog(H,f,A,b,[],[],LB,[],[],opts);
	    
	    
	        %  %  test the obtained classifier on the whole dataset
	        % xx1=[x11,x12,x13];
	        % xx2=[x21,x22,x23];
	    
	    
	        f=zeros(size(u,1),1);
	        for j=1:size(u,1)
	            f(j)=alpha(n1+1);
	            for i=1:n1
	            f(j)=f(j)+alpha(i)*Y(i,i)*gausskernel(xtrain(i,:),u(j,:), p);
	            end
	            f(j)=D1(j)*f(j);
	        end
	    
	        for i=1:size(u,1)
	            if f(i)<0
	                tem12(kk)=tem12(kk)+1;
	            end  
	        end
	     
	    end
	    
	    
	    
	    m = ( 1- mean(tem12)/length(D1) )* 100;
	    % s=std(tem12)/length(D1) * 100
	    % ma= ( 1 - min(tem12)/length(D1) ) * 100
	    % mi= ( 1 - max(tem12)/length(D1) ) * 100
	    array_m(iC, ibeta, ip) = m;
	end
    end
end

[m n] = max(array_m(:));
[iCmax ibetamax ipmax] = ind2sub(size(array_m),n);

logC(iCmax)
logbeta(ibetamax)
logp(ipmax)
m

%save('result1.mat', 'array_m', 'logC', 'logbeta', 'logp', 'm')


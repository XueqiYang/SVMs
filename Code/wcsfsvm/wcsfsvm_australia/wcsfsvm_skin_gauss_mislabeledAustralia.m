clear all; clc;
load('./Australia.mat');
label = Australia(:, 1);
samples = Australia(:, 2:end);
clear Australia

[norm_samples,~] = mapminmax(samples',0,1);
samples = norm_samples';

N=length(label);
K=10;                %repeating time 
% parameters: 
% logC = 5;
% logbeta = 1;
% logp = -5;

logC = -1:1:6;%2;%6;
logbeta = -5:1:2;%-2;%2;
logp = -7:1:1;%-2;%2;
K_cv = 10;    
iC = 1:length(logC);
ibeta = 1:length(logbeta);
ip = 1:length(logp);
rng('default');
          
errAcc = zeros(K, K_cv);
%%%------------------------updated part--------------------------
for kk=1:K
  indices = crossvalind('Kfold',label,K_cv);                                         
  VectorAccuracy = zeros(1,K_cv);   
  for t = 1:K_cv
    T = t + 1;
    if t == K_cv
        T = 1;
    end                                                                   
    test = (indices == t);
    cv = (indices == T);
    train = ((indices ~= t) & (indices ~= T));

    trainset = samples(train,:);
    trainlabel = label(train,1);
    
    % mislabel the data by Sherry--------------
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
    %  added by Sherry -----------
    
    testset = samples(test,:);
    testlabel = label(test,1);
    cvset = samples(cv,:);
    cvlabel = label(cv,1);
    % 10 folds in total   % 1 fold for cvset
    % 1 fold for test set % 8 fold for train set
    bestAcc = 0;
    %using cvset to find out the best parameter for testset  
    tem12=zeros(length(logC),length(logbeta),length(logp)); % accuracy
    
    for iC= 1:length(logC)   
     for ibeta = 1:length(logbeta)
       for ip  = 1:length(logp)
       % beta= 2^-3.4; %-3.4 ;% 0.03; beta % p = 2^-4;
	    C = 2 ^ logC(iC);
	    beta = 2 ^ logbeta(ibeta);
	    p = 2 ^ logp(ip);

	    C2= C;
	    C1 = beta;
  
          % get the training data
	      % tmp_xlabel = label(indices == 1);
	      % xlabel = [ tmp_xlabel(tmp_xlabel == 1, :); tmp_xlabel(tmp_xlabel ~= 1, :)];
	      % xtrain = samples(indices == 1,:);%£¿£¿
	      % xtrain = [ xtrain(tmp_xlabel == 1, :); xtrain(tmp_xlabel ~= 1, :)];
	            
	        % get the test data
	        % ytest = samples(indices ~= 1, :);
	        % ylabel = label(indices ~= 1);
	   
	        % u=samples;
	        % D1 = label;
	        u = cvset;
	        D1 = cvlabel;
	%--------------------------- below------------------     	    
	        n1=size(trainset,1);
	        n0=sum(trainlabel == 1);
	    
	        Y = diag(trainlabel);
            %generate a matrix with trainlabel as the main diagonal
            
	        % svdd for getting the radius R0
	        LB=zeros(n0,1);
	        UB=C*ones(n0,1);
	        Aeq=ones(1,n0);
	        beq=1;
	    
	        f=zeros(n0,1);
	        for i=1:n0
	          f(i)=-gausskernel(trainset(i,:),trainset(i,:), p);
	        end
	    
	        H=zeros(n0);
	        for i=1:n0
	            for j=1:n0
	               H(i,j)=2*gausskernel(trainset(i,:),trainset(j,:), p);
	            end 
	        end
	    
	         opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
	         z1 = quadprog(H,f,[],[],Aeq,beq,LB,UB,[],opts);
	    
	         for i=1:n0
	            if z1(i)>=1e-3
	                temp=0;
	                for j=1:n0
	                temp=temp+z1(j)*gausskernel(trainset(i,:),trainset(j,:), p);
	                end
	               R0=sqrt(gausskernel(trainset(i,:),trainset(i,:), p)-2*temp+0.5*z1'*H*z1); 
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
	          f(i)=-gausskernel(trainset(n0+i,:),trainset(n0+i,:), p);
	        end
	    
	        H=zeros(n2);
	        for i=1:n2
	            for j=1:n2
	               H(i,j)=2*gausskernel(trainset(n0+i,:),trainset(n0+j,:), p);
	            end 
	        end
	   
	         opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
	         z1 = quadprog(H,f,[],[],Aeq,beq,LB,UB,[],opts);
	    
	         for i=1:n2
	            if z1(i)>=1e-3
	                temp=0;
	                for j=1:n2
	                temp=temp+z1(j)*gausskernel(trainset(n0+i,:),trainset(n0+j,:), p);
	                end
	               R2=sqrt(gausskernel(trainset(n0+i,:),trainset(n0+i,:), p)-2*temp+0.5*z1'*H*z1); 
	               break
                end	    
	         end
	    
	         % calculate the distance for each point in the feature space
	    	 d1=zeros(n0,1);
	         temp1=0;
	         for i=1:n0
	            for j=1:n0
	            temp1=temp1+gausskernel(trainset(i,:),trainset(j,:), p);
	            end
	         end
	    
	         for i=1:n0
	    
	             temp2=0;
	             for j=1:n0
	                 temp2=temp2+gausskernel(trainset(i,:),trainset(j,:), p);
	             end
	            d1(i)=sqrt(gausskernel(trainset(i,:),trainset(i,:), p)-2/n0*temp2+1/n0^2*temp1);
	    
	         end
	    
	          d2=zeros(n2,1);
	         temp1=0;
	         for i=1:n2
	            for j=1:n2
	            temp1=temp1+gausskernel(trainset(i+n0,:),trainset(j+n0,:), p);
	            end
	         end
	    
	         for i=1:n2
	             temp2=0;
	             for j=1:n2
	                 temp2=temp2+gausskernel(trainset(i+n0,:),trainset(j+n0,:), p);
	             end
	            d2(i)=sqrt(gausskernel(trainset(i+n0,:),trainset(i+n0,:), p)-2/n2*temp2+1/n2^2*temp1); 
	    
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
	                G(i,j)=gausskernel(trainset(i,:),trainset(j,:), p);
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
%----------------------------------above----------------------------------------	    	    
	        % test the obtained classifier on cv dataset
	       f=zeros(size(u,1),1);
	        for j=1:size(u,1)
	            f(j)=alpha(n1+1); 
	            for i=1:n1
	            f(j)=f(j)+alpha(i)*Y(i,i)*gausskernel(trainset(i,:),u(j,:), p);
	            end
	            f(j)=D1(j)*f(j);
	        end
	    
	        for i=1:size(u,1)
	            if f(i)<0
	                tem12(iC,ibeta,ip)=tem12(iC,ibeta,ip)+1;%error number
                end 
                
            end	     
	    end
	  
	end
    end
      % get the best parameters 	   	    
	    m = ( 1-(tem12)/length(D1) )* 100;
	    % s=std(tem12)/length(D1) * 100
	    % ma= ( 1 - min(tem12)/length(D1) ) * 100
	    % mi= ( 1 - max(tem12)/length(D1) ) * 100
   	    % array_m(iC, ibeta, ip) = m;
        [im, in] = max(m(:));
        [iCmax, ibetamax ipmax] = ind2sub(size(m),in);
        % logC(iCmax)
        % logbeta(ibetamax)
        % logp(ipmax)
        % im
    % then apply the best parameter to test the test set

    tic;%---------------------------------------------------------------
    % test the obtained classifier on test dataset
    C = 2 ^ logC(iCmax);
	beta = 2 ^ logbeta(ibetamax);
	p = 2 ^ logp(ipmax);

	C2= C;
	C1 = beta;

 
    % u=samples;
    % D1 = label;
    u = testset;
    D1 = testlabel;
%-------------------------------------- below------------------     	    
    n1=size(trainset,1);
    n0=sum(trainlabel == 1);

    Y = diag(trainlabel);
    %generate a matrix with trainlabel as the main diagonal
    
    % svdd for getting the radius R0
    LB=zeros(n0,1);
    UB=C*ones(n0,1);
    Aeq=ones(1,n0);
    beq=1;

    f=zeros(n0,1);
    for i=1:n0
      f(i)=-gausskernel(trainset(i,:),trainset(i,:), p);
    end

    H=zeros(n0);
    for i=1:n0
        for j=1:n0
           H(i,j)=2*gausskernel(trainset(i,:),trainset(j,:), p);
        end 
    end

     opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
     z1 = quadprog(H,f,[],[],Aeq,beq,LB,UB,[],opts);

     for i=1:n0
        if z1(i)>=1e-3
            temp=0;
            for j=1:n0
            temp=temp+z1(j)*gausskernel(trainset(i,:),trainset(j,:), p);
            end
           R0=sqrt(gausskernel(trainset(i,:),trainset(i,:), p)-2*temp+0.5*z1'*H*z1); 
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
      f(i)=-gausskernel(trainset(n0+i,:),trainset(n0+i,:), p);
    end

    H=zeros(n2);
    for i=1:n2
        for j=1:n2
           H(i,j)=2*gausskernel(trainset(n0+i,:),trainset(n0+j,:), p);
        end 
    end

     opts = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
     z1 = quadprog(H,f,[],[],Aeq,beq,LB,UB,[],opts);

     for i=1:n2
        if z1(i)>=1e-3
            temp=0;
            for j=1:n2
            temp=temp+z1(j)*gausskernel(trainset(n0+i,:),trainset(n0+j,:), p);
            end
           R2=sqrt(gausskernel(trainset(n0+i,:),trainset(n0+i,:), p)-2*temp+0.5*z1'*H*z1); 
           break
        end	    
     end

     % calculate the distance for each point in the feature space
	 d1=zeros(n0,1);
     temp1=0;
     for i=1:n0
        for j=1:n0
        temp1=temp1+gausskernel(trainset(i,:),trainset(j,:), p);
        end
     end

     for i=1:n0

         temp2=0;
         for j=1:n0
             temp2=temp2+gausskernel(trainset(i,:),trainset(j,:), p);
         end
        d1(i)=sqrt(gausskernel(trainset(i,:),trainset(i,:), p)-2/n0*temp2+1/n0^2*temp1);

     end

      d2=zeros(n2,1);
     temp1=0;
     for i=1:n2
        for j=1:n2
        temp1=temp1+gausskernel(trainset(i+n0,:),trainset(j+n0,:), p);
        end
     end

     for i=1:n2
         temp2=0;
         for j=1:n2
             temp2=temp2+gausskernel(trainset(i+n0,:),trainset(j+n0,:), p);
         end
        d2(i)=sqrt(gausskernel(trainset(i+n0,:),trainset(i+n0,:), p)-2/n2*temp2+1/n2^2*temp1); 

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
            G(i,j)=gausskernel(trainset(i,:),trainset(j,:), p);
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
    %  test the obtained classifier on test dataset
    f=zeros(size(u,1),1);
    for j=1:size(u,1)
        f(j)=alpha(n1+1); 
        for i=1:n1
        f(j)=f(j)+alpha(i)*Y(i,i)*gausskernel(trainset(i,:),u(j,:), p);
        end
        f(j)=D1(j)*f(j);
    end

    for i=1:size(u,1)
        if f(i)<0
            errAcc(kk, t) = errAcc(kk, t)+1;%error number
        end
    end
  times(t)=toc;%-----------------------------------------------------  
  end
AverageTime(kk) = mean(times);
end

meanAcc = mean(1 - mean(errAcc)/size(u,1))
stdAcc = std(1-mean(errAcc/size(u, 1)))
mTime = mean(AverageTime)
maxAcc =max(1- mean((errAcc)/size(u, 1)))
minAcc =min(1- mean((errAcc)/size(u, 1)))

%save('result1.mat', 'array_m', 'logC', 'logbeta', 'logp', 'm')
% quit
dbstop if error

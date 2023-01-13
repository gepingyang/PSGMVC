%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     Code for PNGMVC       %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear;  clear memory; clc;




load('BBCSport.mat'); %loading data

L = 128;    %code length 
m=200;    % number of anchor
p =200;  %number of prototype
N = size(X{1},1); % number of data

viewNum = length(X); %viewNum
iteration =7; %number of iteration

beta =m/N/viewNum;
gamma=1/p/L; 
lambda = 1/p/L/viewNum;

pr = 0;
ac = 0;
nmi = 0;
repetition =10;
X = cellfun(@(x) (x - mean(x, 2)) ./ std(x, 0, 2), X, 'uni', 0);%normalized data
for re = 1:repetition

       rand('seed',re);
     for v = 1:viewNum

        indSmp = randperm(N);
        anchor =X{v}(indSmp(1:m),:);% random select anchor
        dist = EuDist2(X{v},anchor,0);
        sigma = mean(min(dist,[],2).^0.5)*2;
        feaVec = exp(-dist/(2*sigma*sigma));
        E{v} = bsxfun(@minus, feaVec', mean(feaVec',2));% Centered data 
        
    end


    
sel_sample = E{viewNum}(:,randsample(N, m),:);
[pcaW, ~] = eigs(cov(sel_sample'), L);
B = sign(pcaW'* E{viewNum});
U = cell(viewNum,1);
P = B(:,randsample(N, p));
Q =  B'*P;
[~,ind] =max(Q,[],2);
H = sparse(ind,1:N,1,p,N,N);
H = full(H);



alpha = zeros(1,viewNum) + 1 / viewNum;

XXT = cell(viewNum,1);
for v = 1:viewNum
    EET{v}= E{v}* E{v}';
end
Y= Y_Initialize(p,  numel(unique(gt)));

%------------End Initialization--------------



disp('----------The proposed PNGMVC----------');

    
%-----Start optimization---------------------------
for iter = 1:iteration
    fprintf('The %d-th iteration...\n',iter);
    %---------Update U^v--------------

        UX = zeros(L,N);
        for v = 1:viewNum
            
            U{v} = alpha(v)^2*B*E{v}'/(alpha(v)^2*EET{v}+beta*eye(size(E{v},1)));
            UX  = UX + alpha(v)^2*U{v} *E{v};
        end


    %---------Update B--------------
    
    
      
        YY1 = Y./repmat(sqrt(sum(Y.^2,1)),p,1);
        mu = 0.0001;
        grad = - 2*UX   - gamma*P*H ;
        B = sign(B-1/mu*grad); 
        B(B==0) = -1;

    
    %---------update P-----------------
    
        P = sign(B*H'); P(P==0) = 1;
        mmu = 0.001;
        grad = -gamma *B*H' - 2*lambda*P*YY1*YY1';
        P    = sign(P-1/mmu*grad); P(P==0) = 1;
   
    %---------------update H-------------
        Q =  B'*P; 
        [~,indx] = max(Q,[],2);
        H = sparse(indx,1:N,1,p,N,N);
        H = full(H);
    %-------update F------------------
        Y = update_Y(P',Y);


    
    %---------Update alpha--------------
        d = zeros(viewNum, 1); 
        for v = 1:viewNum
            d(v) = norm(B-U{v}*E{v},'fro')^2 ;            
        end
        c = bsxfun(@power,d, 1/2);     
        alpha = bsxfun(@rdivide,c,sum(c));    

end
disp('----------Main Iteration Completed----------');
[~,H] = max(H,[],1);
 Y= vec2ind(Y');
 y_pred = Y(H);
 Result = ClusteringMeasure_new(gt, y_pred);
 ac = ac + Result.ACC; 
 nmi = nmi+ Result.NMI; 
 pr = pr + Result.Purity;



end

fprintf("the average performance as follows:\n ACC:%12.6f\t NMI:%12.6f\t Purity:%12.6f",[ac/repetition nmi/repetition pr/repetition]);


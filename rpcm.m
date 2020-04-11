function [acc,nmi]=rpcm(X, feat, gnd, opts);
modelstyle=opts.modelstyle;
theta=opts.theta;
D=X;
%% Initializing weights of neural networks
weight=initializeweight(X,D,opts);
%%
switch modelstyle    
    case 'rpcm_f2'
         [weight, ~, ~] = rpcm_f2(X,D,weight,opts);
    case 'rpcm_l1'
         [weight, ~, ~, ~] = rpcm_l1(X,D,weight,opts);
    case 'rpcm_unclear'
         %theta=[0.00001,0]; 
         [weight, ~, ~, ~] = rpcm_unclear(X,D,weight,opts);
    case 'rpcm_l1f2'
         [weight, ~, ~, ~] = rpcm_l1f2(X,D,weight,opts);       
end

%Q = orth(A');
%B = A*Q;
%[Z,E] = preinexact_alm_lrr_l21(X,B,opts);
%Z = Q*Z;
%% Coding by fast computing the neural networks
H=actfun(weight,feat,opts.act_fun);
R2=H{end}; 
%% Clustering by lbsc

accplrall=0; 
for i=1:size(theta,2)
%disp(['The truncated threshold:  ' num2str(theta(i))]);
%R = Shrink_L1(R2,theta(i));
if max(max(abs(R2)))>theta(i)
R=R2.*(R2>theta(i))+R2.*(R2<-theta(i));
[accplrall_temp, nmiplrall_temp]=accncutLSC(R',gnd);
end
if accplrall_temp>=accplrall
   accplrall=accplrall_temp;
   nmiplrall=nmiplrall_temp;
end
end
if accplrall==0
   R = ( abs(R2) + abs(R2') ) / 2 ;
   [accplrall, nmiplrall]=accnmi( R, gnd );
end
%disp(['all data acc:  ' num2str(accplrall)]);
%disp(['all data nmi:  ' num2str(nmiplrall)]);
%disp(['coding time:   ' num2str(codingtime)]);
acc=accplrall;
nmi=nmiplrall;
end

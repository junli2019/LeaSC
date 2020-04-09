function [Z,R2]=rpcmcode(X, feat, opts);
modelstyle=opts.modelstyle;
D=X;
%% Initializing weights of neural networks
weight=initializeweight(X,D,opts);
%%
switch modelstyle    
    case 'rpcm_f2'
         [weight, Z] = rpcm_f2(X,D,weight,opts);
    case 'rpcm_l1'
         [weight, Z, ~] = rpcm_l1(X,D,weight,opts);
    case 'rpcm_unclear'
         [weight, Z, ~] = rpcm_unclear(X,D,weight,opts);
    case 'rpcm_l1f2'
         [weight, Z, ~] = rpcm_l1f2(X,D,weight,opts);       
end

%% Coding by fast computing the neural networks
H=actfun(weight,feat,opts.act_fun);
R2=H{end}; 
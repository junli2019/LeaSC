
clear all;
addpath( '.../LeaSC/utilities' ) ;

load anexample800.mat;

opts.epsilon = 0.00001; % learning rate
opts.gamma= 0.00001;    % weight reguleration 
opts.act_fun = 'max';   % activation function
opts.hidnum= [100];    % hiden layers;
opts.NNmaxiter=1;       % the number of training neural network in one iteration
opts.trainIter=45;      % before trainIter, do not train neural network to save training time

%% parameters for RPCM
opts.modelstyle='rpcm_f2';
opts.overlinealpha=1;   
opts.beta=0.5; 
opts.display = 1;  
[Z,P]=rpcmcode(D, X, opts);
imodel=1;
rpcm_code{imodel}.Z=Z;
rpcm_code{imodel}.P=P;
figure
imshow(rpcm_code{imodel}.Z*15)
figure
imshow(rpcm_code{imodel}.P*15)
%%%%%%%%%%%%%%%%%%%%
opts.modelstyle='rpcm_l1'; 
opts.overlinealpha=0.1;   
opts.beta=1; 
opts.display = 1;  
[Z,P]=rpcmcode(D, X, opts);
imodel=2;
rpcm_code{imodel}.Z=Z;
rpcm_code{imodel}.P=P;
figure
imshow(rpcm_code{imodel}.Z*15)
figure
imshow(rpcm_code{imodel}.P*15)
%%%%%%%%%%%%%%%%%%%%
opts.modelstyle='rpcm_unclear'; 
opts.overlinealpha=1;   
opts.beta=1; 
opts.display = 1;  
[Z,P]=rpcmcode(D, X, opts);
imodel=3;
rpcm_code{imodel}.Z=Z;
rpcm_code{imodel}.P=P;
figure
imshow(rpcm_code{imodel}.Z*15)
figure
imshow(rpcm_code{imodel}.P*15)
%%%%%%%%%%%%%%%%%%%%
opts.modelstyle='rpcm_l1f2'; 
opts.overlinealpha=1;  
opts.alpha=0.01;   
opts.beta=1; 
opts.display = 1;  
[Z,P]=rpcmcode(D, X, opts);
imodel=4;
rpcm_code{imodel}.Z=Z;
rpcm_code{imodel}.P=P;
figure
imshow(rpcm_code{imodel}.Z*15)
figure
imshow(rpcm_code{imodel}.P*15)

% save example800image sc_code rpcm_code;

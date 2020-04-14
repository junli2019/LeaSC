clear all;
addpath( '.../LeaSC/utilities' ) ;

%load MNIST_SCALL; 
% Before you run this code, you need to extract features of MNIST as follows.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%     extract features of MNIST    %%%%%%%%%%%%%%%%
%   = Download scattering transform package ScatNet (v0.2) from
%     http://www.di.ens.fr/data/software/
%     and install.
%   = Download MATLAB Code for SSC-OMP from
%     http://www.vision.jhu.edu/code/
%     and run the code of Load data in run_MNIST.m

%     feat=MNIST_SC_DATA;  %  3473*70000
%     gnd=MNIST_LABEL+1;   %  70000*1      [1-10]


%% parameters for neural networks
opts.epsilon = 0.00001; % learning rate
opts.gamma= 0.00001;    % weight reguleration 
opts.act_fun = 'max';   % activation function
opts.hidnum= [1500];    % hiden layers;
opts.NNmaxiter=1;       % the number of training neural network in one iteration
opts.trainIter=45;      % before trainIter, do not train neural network to save training time
%% parameters for PCM
model.set{1}='rpcm_f2';
model.set{2}='rpcm_l1';
model.set{3}='rpcm_unclear';
model.set{4}='rpcm_l1f2'; % alpha=100, beta=10;

opts.overlinealpha=1;  % overlinealpha=1, beta=0.01 for 'cpcm_unclear'
opts.beta=0.001;       % beta for the noise
opts.display = 1;      % print the loss in each iteration 
opts.choosenum=50;     % choosing the number of samples
opts.choosestyle = 1;  % style==1  randomly select the selectnum samples 
                       %           from the data of every clustering center
                       % style==0  randomly select the selectnum samples
                       %           from the whole data
opts.theta=[0.005, 0.001,0.0005,0.0001,0.00005,0]; 
opts.repeatnum=1;
opts.choosenum=50;
repeatnum=opts.repeatnum;
choosenum=opts.choosenum;
choosestyle=opts.choosestyle;
betaset=[0.01 0.1 0.1 10]; 

ACC=zeros(4,repeatnum);
NMI=zeros(4,repeatnum);
for imodel=1:length(model.set)   
    disp(['---------model = : ' num2str(model.set{imodel}) '---------' ]);    
    opts.modelstyle=model.set{imodel};
    opts.beta=betaset(imodel); % 

    if imodel==4
       opts.alpha=100; 
    end

    for ii=1:repeatnum
        [X, ~]=selectsrno(feat,gnd',choosenum,choosestyle);    
        [acc,nmi]=rpcm(X, feat, gnd, opts);
        disp(['beta = : ' num2str(opts.beta) '  repeatnum = :   ' num2str(ii)]);
        disp(['acc = : ' num2str(acc) '  nmi = : ' num2str(nmi)]);
        ACC(imodel,ii)=acc;
        NMI(imodel,ii)=nmi;
    end
    %savefilename=['mnst-sc-beta-', opts.modelstyle];
    %save(savefilename, 'ACC', 'NMI', 'betaset','opts');
end


function [weight, Z, E, err]=rpcm_l1f2(X,D,weight,opts)
%--------------------------------------------------------------------------
% Copyright @ Jun Li, 2018
%--------------------------------------------------------------------------

% Input
% X                              Data matrix, d * n
% D                              Dictionary, d * m
% overlinealpha, alpha, beta     parameters, overlinealpha, alpha, beta>0

% This routine uses ADMM algorithm to solve the following PCM with l1-norm optimization problem:
% min |Z-f(weight,X)|^2+overlinealpha*|Z|_1+alpha*|Z|_F^2+beta*|E|_1
% s.t., X = AZ+E

%Output
% Z              representation matrix m*n         
% E              sprse noise matrix d*n  
% weight         training weights

alpha=opts.alpha;         %      parameters
overlinealpha=opts.overlinealpha;  %      parameters
beta=opts.beta;          %      parameters
display=opts.display;  
tol = 1e-3;
maxIter = 1e6;
[d n] = size(X);
m = size(D,2);
rho = 1.1;
max_mu = 1e10;

mu = 0.1;
atx = D'*X;

trainIter=opts.trainIter;
%% Initializing optimization variables
% intialize
J = zeros(m,n);
Z = zeros(m,n);
E = sparse(d,n);

Q1 = zeros(d,n);
Q2 = zeros(m,n);

%% Start main loop
iter = 0;
allstop=[];
while iter<maxIter
    iter = iter + 1;
    %update J
    temp = Z + Q2/mu;
    J = Shrink_L1(temp,overlinealpha/mu);
    J = J - diag(diag(J));
    if (iter>trainIter) && (iter<maxIter)
       [weight,J,~]=Learnmap(weight,J,X,opts);
    end
    %udpate Z
    inv_a = inv(D'*D+(1+alpha/mu)*eye(m));
    Z = inv_a*(atx-D'*E+J+(D'*Q1-Q2)/mu);
    %update E
    xmaz = X-D*Z;
    temp = xmaz+Q1/mu;
    %E = Shrink_L1(temp,beta/mu);
    E = solve_l1l2(temp,beta/mu);

    leq1 = xmaz-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    allstop=[allstop stopC];
    if display && (iter==1 || mod(iter,10)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Q1 = Q1 + mu*leq1;
        Q2 = Q2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end
[weight,~,~]=Learnmap(weight,Z,X,opts);
err=allstop;
end
function [weight, Z, E, err]=rpcm_unclear(X,D,weight,opts)
%--------------------------------------------------------------------------
% Copyright @ Jun Li, 2018
%--------------------------------------------------------------------------

% Input
% X                        Data matrix, d * n
% D                        Dictionary, d * m
% overlinealpha, beta      parameters, overlinealpha, beta>0

% This routine uses ADMM algorithm to solve the following PCM with nuclear-norm optimization problem:
% min |Z-f(weight,X)|^2+overlinealpha*|Z|_*+beta*|E|_2,1
% s.t., X = AZ+E

% Output
% Z              representation matrix m*n         
% E              sprse noise matrix d*n  
% weight         training weights

overlinealpha=opts.overlinealpha;         %      parameters
beta=opts.beta;          %      parameters
display=opts.display;  
tol = 1e-3;
maxIter = 1e6;
[d n] = size(X);
m = size(D,2);
rho = 1.1;
max_mu = 1e10;
mu = 1e-2;
atx = D'*X;
inv_a = inv(D'*D+eye(m));
trainIter=50;
%% Initializing optimization variables
% intialize
J = zeros(m,n);
Z = zeros(m,n);
E = sparse(d,n);

Q1 = zeros(d,n);
Q2 = zeros(m,n);

%% Start main loop
iter = 0;
if display
    disp(['initial,rank=' num2str(rank(Z))]);
end
    allstop=[];
while iter<maxIter
    iter = iter + 1;
    %update J
    temp = Z + Q2/mu;
    [J] = solve_unclear(temp,overlinealpha/mu);
    if (iter>trainIter) && (iter<maxIter)
       [weight,J,~]=Learnmap(weight,J,X,opts);
    end
    %udpate Z
    Z = inv_a*(atx-D'*E+J+(D'*Q1-Q2)/mu);
    %update E
    xmaz = X-D*Z;
    temp = xmaz+Q1/mu;
    E = solve_l1l2(temp,beta/mu);
    
    leq1 = xmaz-E;
    leq2 = Z-J;
    
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    allstop=[allstop stopC];
    if display && (iter==1 || mod(iter,10)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Q1 = Q1 + mu*leq1;
        Q2 = Q2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end
[weight,J,~]=Learnmap(weight,J,X,opts);
err=allstop;
end










% 
% 
% alpha=opts.alpha;         %      parameters
% beta=opts.beta;          %      parameters
% display=opts.display;  
% alpha=10;
% beta=10; 
% gamma=1;
% tol = 1e-9;
% maxIter = 1e6;
% trainIter=maxIter;
% [d,n] = size(X);
% m = size(D,2);
% rho = 1.1;
% max_mu = 1e10;
% mu = 1e-5;
% %dtx = D'*X;
% dtd=D'*D;
% %% Initializing optimization variables
% % intialize
% J = zeros(m,n);
% Z = zeros(m,n);
% E = sparse(d,n);
% Q = zeros(m,n);
% %% Start main loop
% iter = 0;
% if display
%    disp(['initial,rank=' num2str(rank(Z))]);
% end
% while iter<maxIter
%     iter = iter + 1;
%     %update J
%     temp = Z + Q/mu;
%     J = solve_unclear(temp,alpha/mu); 
%     %udpate Z
%     Z = inv(gamma*dtd+mu*eye(m))*(gamma*D'*(X-E)+mu*J-Q/mu);
%     % training neural networks
%     if (iter>trainIter) && (iter<maxIter)
%        [weight,Z,~]=Learnmap(weight,Z,X,opts);
%     end
%     %update E
%     temp = X-D*Z;
%     E = solve_l1l2(temp,beta/gamma);
%     
%     res=temp-E;
%     stopres = max(max(abs(res)));
%     if display && (iter==1 || mod(iter,10)==0 || stopres<tol)
%         disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%             ',rank=' num2str(rank(Z,1e-4*norm(Z,2))) ',stopres=' num2str(stopres,'%2.3e')]);
%     end
%     if stopres<tol 
%         break;
%     else
%        Q = Q + mu*(Z-J);
%        mu = min(max_mu,mu*rho);
% 
%     end
% end
% [weight,Z,~]=Learnmap(weight,Z,X,opts);
% end


















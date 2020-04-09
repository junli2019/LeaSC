function [weight,Z,err]=rpcm_f2(X,D,weight,opts)
%--------------------------------------------------------------------------
% Copyright @ Jun Li, 2018
%--------------------------------------------------------------------------

% Input
% X                      Data matrix, dim * num
% overlinealpha, beta    parameter, overlinealpha, beta>0


% Output the solution to the following problem:
% min |Z-f(weight,X)|^2+overlinealpha||Z||_F^2+beta*||X-DZ||_F^2
overlinealpha=opts.overlinealpha;
beta=opts.beta;
[num] = size(X,2);
% for i = 1 : num
%    X(:,i) = X(:,i) / norm(X(:,i)) ; 
% end
%% l2 coding
dtx=D'*X;
dtd=D'*D;
I = (overlinealpha/beta)* eye(num);
Z = (dtd+I) \( dtx);
% numtest = size(Z,2) ;
% for i = 1 : numtest
%    Z(:,i) = Z(:,i) / norm(Z(:,i)); 
% end

%% training neural networks
disp(['training neural networks----- start ']);
[weight,Zlearn,err]=Learnmap(weight,Z,X,opts);
disp(['training neural networks----- end ']);

end
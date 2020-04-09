function Z = solve_lsr( X , D, opts)

%--------------------------------------------------------------------------
% Copyright @ Jun Li, 2018
%--------------------------------------------------------------------------

% Input
% X             Data matrix, dim * num
% D             Dictionary, dim * dnum
% lambda        parameter, lambda>0


% Output the solution to the following problem:
% min ||X-DZ||_F^2+lambda||Z||_F^2

% Z             dnum * num

if nargin < 3
    lambda = 0.004 ;
else
    lambda=opts.lambda;
end
dnum = size(D,2) ;

% for i = 1 : num
%    X(:,i) = X(:,i) / norm(X(:,i)) ; 
% end

Z = (D'*D+lambda * eye(dnum)) \ D' * X ;
end
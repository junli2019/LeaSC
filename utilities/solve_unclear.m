function [Z] = solve_unclear(Z,lambda) 
%--------------------------------------------------------------------------
% Copyright @ Jun Li, 2018
%--------------------------------------------------------------------------

% This routine uses SVD to reduce the rank of the matrix Z:
[U,sigma,V] = svd(Z,'econ');
sigma = diag(sigma);
svp = length(find(sigma>lambda));
if svp>=1
   sigma = sigma(1:svp)-lambda; 
else
   svp = 1;
   sigma = 0;
end
Z = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end
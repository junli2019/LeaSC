function [AC,MIhat]=accncutLSC(Z, gnd)

opt.r = 3;
opt.p = 500;
opt.kmMaxIter = 3;
%opt.mode='random';
nCluster=length( unique( gnd ) ) ;

% numtest = size(Z,1) ;
% for i = 1 : numtest
%    Z(i,:) = Z(i,:)./ norm(Z(i,:)) ; 
% end
% Z(~isnan(Z))=0;

%rand('twister',5489) 
res = LSC(Z, nCluster, opt);
%Elapsed time is 15.471343 seconds.
%tic;res=litekmeans(Z,nCluster,'MaxIter',100,'Replicates',10);toc
% gnd=gnd+1;
% res=res-1;

res = bestMap(gnd,res);
AC = length(find(gnd == res))/length(gnd);
MIhat = MutualInfo(gnd,res);
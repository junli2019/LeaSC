function [ acc, nmi ] = accnmi( Z, s )
%ACCNMI Summary of this function goes here
%   Detailed explanation goes here
rho=0.7;
nCluster = length( unique( s ) ) ;
CKSym = BuildAdjacency(thrC(Z,rho));
grps = SpectralClustering(CKSym,nCluster);
grps = bestMap(s,grps);
missrate = sum(s(:) ~= grps(:)) / length(s);
acc=1-missrate;
nmi = MutualInfo(s,grps);
end


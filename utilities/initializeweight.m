function weight=initializeweight(X,D,opts)
%% Initializing weights
d=size(X,1);
m=size(D,2);
hidnum=[d opts.hidnum m];
for ii=2:size(hidnum,2)
weight{ii}=0.01*randn(hidnum(ii),hidnum(ii-1));
end
end
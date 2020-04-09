function grad=gradfun(W,H,Er,opts)
n=numel(W);
num=size(Er,2);
grad{n}=W{n}'*Er;
if n-1>=2
for i=n-1:2
dH{i}=dfun(H{i},opts.act_fun);
grad{i}=(grad{i+1}.*dH{i})*H{i-1}'./num+opts.gamma*W{i};
end
end
end
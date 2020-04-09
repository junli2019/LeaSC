function [W,HH,err]=Learnmap(W,ZZ,X,opts)
n=numel(W);
ntop=size(W{n},2);
H=actfun(W,X,opts.act_fun);
Er=ZZ-H{n};
err=[];
err=[err mean(mean(abs(Er)))];
W{n}=ZZ*H{n-1}'*inv(H{n-1}*H{n-1}'+0.001*eye(ntop));
for i=1:opts.NNmaxiter
    % forwrd
    H=actfun(W,X,opts.act_fun);
    % computing Weights of top layer
    W{n}=ZZ*H{n-1}'*inv(H{n-1}*H{n-1}'+opts.gamma*eye(ntop));
    % computing W1
    Er=ZZ-H{n};
    grad=gradfun(W,H,Er,opts);
    if n-1>=2
    for ii=n-1:2
        W{ii}=W{ii}+opts.epsilon*(grad{ii});
    end
    end
    objerr=max(max(abs(Er)));
    err=[err mean(mean(abs(Er)))];
   disp(['Object error  ' num2str(objerr)]);
end
HH=H{n};
end
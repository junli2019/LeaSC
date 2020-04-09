function [H]=actfun(W,X,act_fun)
H{1}=X;
n=numel(W);
for i=2:n-1
   switch act_fun 
          case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                H{i} = sigm(W{i}*H{i-1});
          case 'tanh'
                H{i} = tanh(W{i}*H{i-1});
          case 'max'
                H{i} = max(0,W{i}*H{i-1});
          case 'shrink'
                H1=W{i}*H{i-1};
                theta=0.1;
                H{i} =(((-theta<H1)+(H1<theta))<1.5).* H1;
          case 'softplus'
                H{i} = log(1+exp(W{i}*H{i-1}));
          case 'linear'
                H{i} = W{i}*H{i-1};
   end
end
   H{n} = W{n}*H{n-1};
end

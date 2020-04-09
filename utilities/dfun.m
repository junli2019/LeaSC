function deltaH=dfun(H,activation_function)
   switch activation_function 
          case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                deltaH=H.*(1-H);
          case 'tanh'
                deltaH=(1-H).^2;
          case 'max'
                deltaH=(+(H>0));
           case 'shrink'
                deltaH= (H~=0);
          case 'softplus'
                deltaH= exp(H)./(1+exp(H));
          case 'linear'
               deltaH=ones(size(H));
   end

end
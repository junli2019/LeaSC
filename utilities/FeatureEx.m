function dat = FeatureEx(DATA, nDim)

% eigenface extracting
if nDim < size(DATA, 1)
    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(DATA,nDim);
    dat  =  disc_set'*DATA;
else
    dat = DATA;
end;

dat  =  dat./( repmat(sqrt(sum(dat.*dat)), [size(dat, 1),1]) );

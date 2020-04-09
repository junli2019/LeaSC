function [train_d, train_t]=selectsrno(featureMat,labelMat,selectnum,style)

% style==0  randomly select the selectnum samples
%           from the whole data
if style==0
   Y=randperm(size(labelMat,2));
   Ytrain=sort(Y(1:selectnum));
   train_d=featureMat(:,Ytrain);
   train_t=labelMat(:,Ytrain);
end

% style==1  randomly select the selectnum samples 
%           from the data of every clustering center
if style==1
labelnum=max(labelMat);
num=[];
for ii=1:labelnum
    num=[num sum(labelMat==ii)];
end

train_d=[];
train_t=[];
allnum=0;
for i=1:labelnum
    feature=featureMat(:,allnum+1:allnum+num(i));
    label = labelMat(:,allnum+1:allnum+num(i));
    kk = randperm(num(i));
    train_d=[train_d feature(:,kk(1:selectnum))];
    train_t=[train_t label(:,kk(1:selectnum))];
    allnum=allnum+num(i);
end
end

end
load('armdata.mat')
load('test_data.mat')

ttt=10;
batch=size(armdata,1)/ttt;
number_of_redundant=5;
k=30;
k2=5;
train=0;

if train==1
    Output=cell(1,ttt);
    tic
    for i=1:ttt
        training=batch*i;
        Output{i}=TRR(armdata(1:training,3:4),armdata(1:training,1:2),test_data(:,3:4),number_of_redundant,k,k2);
        
    end
    toc
else
    load('output.mat')
end

trueT=zeros(1,ttt);
error=zeros(ttt,size(Output{1},2));
for i=1:ttt
    true=1;
    %figure;hold on;
    for ii=1:size(Output{i},2)
        if size(Output{i}{ii},1)<3
            true=true+1;
        end
        A=Output{i}{ii}-test_data(ii,1:2).*ones(size(Output{i}{ii}));
        error(i,ii)=min((A(:,1)-A(:,2)).^2);
        %scatter(Output{i}{ii}(:,1),Output{i}{ii}(:,2));
    end
    trueT(1,i)=true/size(Output{i},2);
end

errorave=sum(error,2)/size(Output{i},2);



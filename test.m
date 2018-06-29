
%% 2-dimension
times=50;
num_gradient=1000;
errorthreshold=0.1;
plotf=10;

Nt=20;
k=30;
k2=5;
grids=0;

sita1=[-pi/4 pi/4];
sita2=[-pi pi];
fai1=[0 0];

l1=10;
l2=3;
number_of_redundant=5;
intervals=10;
input1=sita1(1,1):(sita1(1,2)-sita1(1,1))/intervals:sita1(1,2);
input2=sita2(1,1):(sita2(1,2)-sita2(1,1))/intervals:sita2(1,2);

% test point
c=1;
test_input=[];
for i=1:size(input1,2)
    for ii=1:size(input2,2)
        test_input(c,:)=[input1(1,i),input2(1,ii)];
        c=c+1;
    end
end

test_output=redundant_arm_dim2(test_input,l1,l2);

errorave=zeros(times,1);
errorvar=zeros(times,1);
errorAveknn=zeros(times,1);
errorvarknn=zeros(times,1);
errorAveknn2=zeros(times,1);
errorvarknn2=zeros(times,1);
errorAve5=zeros(times,1);
errorvar5=zeros(times,1);

testtimepro=zeros(times,1);
testtimeknn=zeros(times,1);
testtimeknn2=zeros(times,1);
testtimellm=zeros(times,1);
traintimellm=zeros(times,1);
traintime=zeros(times,1);

errorbar11=zeros(times,1);
errorbar12=zeros(times,1);
gooderror=zeros(times,1);
gooderrorrate=zeros(times,1);

errorbar21=zeros(times,1);
errorbar22=zeros(times,1);
gooderror2=zeros(times,1);
gooderrorrate2=zeros(times,1);

errorbar31=zeros(times,1);
errorbar32=zeros(times,1);
gooderror3=zeros(times,1);
gooderrorrate3=zeros(times,1);

errorbar51=zeros(times,1);
errorbar52=zeros(times,1);
gooderror5=zeros(times,1);
gooderrorrate5=zeros(times,1);

True_answerNum=zeros(times,1);
Wrong_answerNum=zeros(times,1);
True_answerrate=zeros(times,1);

input=[];
output=[];
Cinput=[];

for t=1:times

    N=num_gradient;

    tinput=zeros(N,2);
    tinput(:,1)=ones(N,1)*sita1(1,1)+(sita1(1,2)-sita1(1,1))*rand(N,1);
    tinput(:,2)=ones(N,1)*sita2(1,1)+(sita2(1,2)-sita2(1,1))*rand(N,1);
    toutput=redundant_arm_dim2(tinput,l1,l2);
    
    %% f| R^2 -> circle*R^1
    tCinput(:,1)=cos(tinput(:,2));
    tCinput(:,2)=sin(tinput(:,2));
    tCinput(:,3)=tinput(:,1);
    tic
    input=[input;tinput];
    output=[output;toutput];
    traintime(t,1)=toc;
    Cinput=[Cinput;tCinput];
    
           
    %% simple knn
    error3=zeros(size(test_output,1),1);
    
    database=output;
    Input2=zeros(size(test_output,1),2);
    Output2=zeros(size(test_output,1),2);
    tic
    for i=1:size(test_output,1)
        
        indexes = zeros(k2, 1);
        sqDists = zeros(k2, 1);
        D = sum(((database - repmat(test_output(i,:), size(database, 1), 1)).^2), 2);
        for l = 1:k2
            [sqDists(l), indexes(l)] = min(D);
            D(indexes(l)) = inf;
        end
        
        % INVERSE DISTANCE WEIGHT
        Input2(i,1)=IDW2(database(indexes,:),input(indexes,1),test_output(i,:),length(indexes),-2);
        Input2(i,2)=IDW2(database(indexes,:),input(indexes,2),test_output(i,:),length(indexes),-2);
        Input5(i,1)=IDW(database(indexes,1),database(indexes,2),input(indexes,1),test_output(i,1),test_output(i,2),-2,'ng',length(indexes));
        Input5(i,2)=IDW(database(indexes,1),database(indexes,2),input(indexes,2),test_output(i,1),test_output(i,2),-2,'ng',length(indexes));
        
        Output2(i,:)=redundant_arm_dim2(Input2(i,:),l1,l2);
        error3(i,1)=norm(Output2(i,:)-test_output(i,:));
        if error3(i,1)<errorthreshold
            gooderror2(t,1)=gooderror2(t,1)+1;
        end
    end
    testtimeknn(t,1)=toc;
    errorvarknn(t,1)=var(error3);
    errorAveknn(t,1)=sum(error3)/size(error3,1);
    gooderrorrate2(t,1)=gooderror2(t,1)/size(error3,1);
    errorbar21(t,1)=max(error3)-errorAveknn(t,1);
    errorbar22(t,1)=errorAveknn(t,1)-min(error3); 
    
   

end

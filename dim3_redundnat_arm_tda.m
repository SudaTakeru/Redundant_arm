addpath('learner')

%% 3-dimension
times=20;
num_gradient=5000;
errorthreshold=0.1;
plotf=20;
pd =10;
Nt=20;
k=20;
k2=5;
grids=0;

sita1=[-pi/4 pi/4];
sita2=[-pi pi];
sita3=[0 pi];
fai1=[0 0];

l1=10;
l2=3;
number_of_redundant=5;
intervals=10;
input1=sita1(1,1):(sita1(1,2)-sita1(1,1))/intervals:sita1(1,2);
input2=sita2(1,1):(sita2(1,2)-sita2(1,1))/intervals:sita2(1,2);
input3=sita3(1,1):(sita3(1,2)-sita3(1,1))/intervals:sita3(1,2);

% test point
c=1;
test_input=[];
for i=1:size(input1,2)
    for ii=1:size(input2,2)
        for iii=1:size(input3,2)
            test_input(c,:)=[input1(1,i),input2(1,ii),input3(1,iii)];
            c=c+1;
        end
    end
end

test_output=redundant_arm_dim3(test_input,l1,l2);

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

    tinput=zeros(N,3);
    tinput(:,1)=ones(N,1)*sita1(1,1)+(sita1(1,2)-sita1(1,1))*rand(N,1);
    tinput(:,2)=ones(N,1)*sita2(1,1)+(sita2(1,2)-sita2(1,1))*rand(N,1);
    tinput(:,3)=ones(N,1)*sita3(1,1)+(sita3(1,2)-sita3(1,1))*rand(N,1);
    toutput=redundant_arm_dim3(tinput,l1,l2);
    
    %% f| R^2 -> circle*R^1 kokomada
    tCinput(:,1)=cos(tinput(:,2));
    tCinput(:,2)=sin(tinput(:,2));
    tCinput(:,3)=tinput(:,1);
    tCinput(:,4)=tinput(:,3);
    tic
    input=[input;tinput];
    output=[output;toutput];
    traintime(t,1)=toc;
    Cinput=[Cinput;tCinput];
    
    %% proposed
    tic
    %[Cinputhat,COut,Cin]=TRR2(output,Cinput,test_output,number_of_redundant,k,k2);
    [Cinputhat,~,COut,Cin]=TRR3(output,Cinput,test_output,number_of_redundant,k,k2,pd);
    testtimepro(t,1)=toc;
    inputhat=cell(size(Cinputhat));
    
    % inveres f
    for i=1:size(Cinputhat,2)
        inputhat{i}(:,1)=Cinputhat{i}(:,3);
        inputhat{i}(:,3)=Cinputhat{i}(:,4);
        for j=1:size(Cinputhat{i},1)
            if Cinputhat{i}(j,2)>=0
                inputhat{i}(j,2)=acos(Cinputhat{i}(j,1));
            else
                inputhat{i}(j,2)=-1*acos(Cinputhat{i}(j,1));
            end
        end
    end
    Cin2=cell(size(Cin));
    for i=1:size(Cin,2)
        for ii=1:size(Cin{i},2)
            Cin2{i}{ii}(:,1)=Cin{i}{ii}(:,3);
            for j=1:size(Cin{i}{ii},1)
                if Cin{i}{ii}(j,2)>=0
                    Cin2{i}{ii}(j,2)=acos(Cin{i}{ii}(j,1));
                else
                    Cin2{i}{ii}(j,2)=-1*acos(Cin{i}{ii}(j,1));
                end
            end
        end
    end
    
    % show result
    
    error=[];
    errore=cell(size(inputhat,2),1);
    i2=1;
    for i=1:size(inputhat,2)
        signal=inputhat{i};
        outputhat=redundant_arm_dim3(signal,l1,l2);
        if test_input(i,1)<0 && test_input(i,2) >0.3 && test_input(i,2) <2.8
            if size(outputhat,1)==2
               True_answerNum(t,1)=True_answerNum(t,1)+1;
            else
               Wrong_answerNum(t,1)=Wrong_answerNum(t,1)+1;
            end
        elseif (test_input(i,1)<-0.4 && test_input(i,2) >-2.6 && test_input(i,2) <-1.1) ...
                ||(test_input(i,1)>0.4 && test_input(i,2) <2.6 && test_input(i,2) >1.1)
            if size(outputhat,1)==1
               True_answerNum(t,1)=True_answerNum(t,1)+1;
            else
               Wrong_answerNum(t,1)=Wrong_answerNum(t,1)+1;
            end
        end
        if (i==14 || i==43 || i==45 ) && mod(t,plotf)==0
            if i2==1
                figure;
            end
            subplot(3,2,2*i2-1)
            hold on;
            scatter(output(:,1),output(:,2),'.')
            scatter(test_output(i,1),test_output(i,2),'k')
            scatter(outputhat(:,1),outputhat(:,2),'*','r')
            for ii=1:size(COut{i},2)
                scatter(COut{i}{ii}(:,1),COut{i}{ii}(:,2),'.');
            end
            legend('data','test','estimate');
            title(['Output space, number of answers = ' num2str(size(outputhat,1))]);
            subplot(3,2,2*i2)
            hold on;
            scatter(input(:,1),input(:,2),'.')
            scatter(inputhat{i}(:,1),inputhat{i}(:,2),'*','r')
            scatter(test_input(i,1),test_input(i,2),'k')
            for ii=1:size(Cin2{i},2)
                scatter(Cin2{i}{ii}(:,1),Cin2{i}{ii}(:,2),'.');
            end
            legend('data','estimate','true');
            title(['input space, numer of training data'  num2str(N*t)]);
            i2=i2+1;
        end
        for ij=1:size(outputhat,1)
            errore{i}(ij,:)=norm((test_output(i,:)-outputhat(ij,:)));
        end
        for ij=1:size(errore{i})
            if errore{i}(ij,1)<errorthreshold
                gooderror(t,1)=gooderror(t,1)+1;
            end
        end
        error=[error;errore{i}];
        
        True_answerrate(t,1)=True_answerNum(t,1)/(Wrong_answerNum(t,1)+True_answerNum(t,1));
    end
    
    
    errorvar(t,1)=var(error);
    errorave(t,1)=sum(error)/size(error,1);
    gooderrorrate(t,1)=gooderror(t,1)/size(error,1);
    errorbar11(t,1)=max(error)-errorave(t,1);
    errorbar12(t,1)=errorave(t,1)-min(error);
                
    %% simple knn
    
    error3=zeros(size(test_output,1),1);
    
    database=output;
    Input2=zeros(size(test_output,1),3);
    Output2=zeros(size(test_output,1),3);
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
        for dim=1:size(Input2,2)
            Input2(i,dim)=IDW2(database(indexes,:),input(indexes,dim),test_output(i,:),k2,-2);
        end
        Output2(i,:)=redundant_arm_dim3(Input2(i,:),l1,l2);
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
    
    %% two knn
    error4=zeros(size(test_output,1),1);
    
    database=output;
    input4=zeros(size(test_output,1),3);
    Output4=zeros(size(test_output,1),3);
    tic
    for i=1:size(test_output,1)
        % Nearest
        indexes1 = zeros(1, 1);
        sqDists1 = zeros(1, 1);
        D = sum(((database - repmat(test_output(i,:), size(database, 1), 1)).^2), 2);
        for l = 1
            [sqDists1(l), indexes1(l)] = min(D);
            D(indexes1(l)) = inf;
        end
        % input space
        sig=input(indexes1,:);
        indexes2 = zeros(k, 1);
        sqDists2 = zeros(k, 1);
        inputdatabase=input;
        inputdatabase(indexes1,:)=[];
        D = sum(((inputdatabase - repmat(sig, size(inputdatabase, 1), 1)).^2), 2);
        for l = 1:k
            [sqDists2(l), indexes2(l)] = min(D);
            D(indexes2(l)) = inf;
        end
        % output space
        indexes3 = zeros(k2, 1);
        sqDists = zeros(k2, 1);
        Odatabase=database;
        Odatabase(indexes1,:)=[];
        Odatabase2=Odatabase(indexes2,:);
        kinput=inputdatabase(indexes2,:);
        D = sum(((Odatabase2 - repmat(test_output(i,:), size(Odatabase2, 1), 1)).^2), 2);
        for l = 1:k2
            [sqDists(l), indexes3(l)] = min(D);
            D(indexes3(l)) = inf;
        end
        
        % INVERSE DISTANCE WEIGHT
        for dim=1:size(input4,2)
            input4(i,dim)=IDW2(Odatabase2(indexes3,:),kinput(indexes3,dim),test_output(i,:),k2,-2);
        end
        Output4(i,:)=redundant_arm_dim3(input4(i,:),l1,l2);
        error4(i,1)=norm(Output4(i,:)-test_output(i,:));
        if error4(i,1)<errorthreshold
            gooderror3(t,1)=gooderror3(t,1)+1;
        end
    end
    testtimeknn2(t,1)=toc;
    errorvarknn2(t,1)=var(error4);
    errorAveknn2(t,1)=sum(error4)/size(error4,1);
    gooderrorrate3(t,1)=gooderror3(t,1)/size(error4,1);
    errorbar31(t,1)=max(error4)-errorAveknn2(t,1);
    errorbar32(t,1)=errorAveknn2(t,1)-min(error4);     
    
    %% LLM
    %{
    error5=zeros(size(test_output,1),1);
    din = 2;
    dout = 2;
    lspecs = {'class', 'LLM'; 'lrate', 1.0; 'radius', 0.3; 'softmax', 1};
    cmd = ['learner = ' lspecs{1,2} '(din, dout, lspecs);'];
    eval(cmd);
    
    learner.init(output,input);
    tic
    for i=1:size(input,1)
        learner.train(output(i,:),input(i,:), 1);
    end
    traintimellm(t,1)=toc;
    inputhatllm=zeros(size(test_output,1),2);
    outputhatllm=zeros(size(test_output,1),2);
    tic
    for i=1:size(test_output,1)
        inputhatllm(i,:)=learner.apply(test_output(i,:));
        outputhatllm(i,:)=redundant_arm_dim2(inputhatllm(i,:),l1,l2);
        
        error5(i,1)=norm(outputhatllm(i,:)-test_output(i,:));
        if error5(i,1)<errorthreshold
            gooderror5(t,1)=gooderror5(t,1)+1;
        end 
    end
    testtimellm(t,1)=toc;
    errorvar5(t,1)=var(error5);
    gooderrorrate5(t,1)=gooderror5(t,1)/size(test_output,1);
    errorAve5(t,1)=sum(error5)/size(error5,1);
    errorbar51(t,1)=max(error5)-errorAve5(t,1);
    errorbar52(t,1)=errorAve5(t,1)-min(error5);
    %}
    %}
end

axis=1*num_gradient:num_gradient:num_gradient*times;
figure;hold on;
plot(axis,errorave);
plot(axis,errorAveknn);
plot(axis,errorAveknn2);
%plot(axis,errorAve5);
title('errorAverage');
legend('proposed','knn','knn2','LLM');

figure;hold on;
plot(axis,gooderrorrate);
plot(axis,gooderrorrate2);
plot(axis,gooderrorrate3);
%plot(axis,gooderrorrate5);
title('under threshold error percentage');
legend('proposed','knn','knn2','LLM');

figure;hold on;
errorbar(axis,gooderrorrate,errorbar12-errorbar11);
errorbar(axis,gooderrorrate2,errorbar22-errorbar21);
errorbar(axis,gooderrorrate3,errorbar32-errorbar31);
%errorbar(axis,gooderrorrate5,errorbar52-errorbar51);
title('under threshold error percentage');
legend('proposed','knn','knn2','LLM');

figure;hold on;
errorbar(axis,errorave,errorbar12-errorbar11);
errorbar(axis,errorAveknn,errorbar22-errorbar21);
errorbar(axis,errorAveknn2,errorbar32-errorbar31);
%errorbar(axis,errorAve5,errorbar52-errorbar51);
title('errorAverage');
legend('proposed','knn','knn2','LLM');

figure;hold on;
plot(axis,errorvar);
plot(axis,errorvarknn);
plot(axis,errorvarknn2);
%plot(axis,errorvar5);
title('errorVariance');
legend('proposed','knn','knn2','LLM');
%{
figure;
plot(axis,True_answerrate);
title('True rate of number of answer');
%}
figure;hold on;
plot(axis,testtimepro);
plot(axis,testtimeknn);
plot(axis,testtimeknn2);
%plot(axis,testtimellm);
title('Test Time');
legend('proposed','knn','knn2','LLM');
%{
figure;hold on;
plot(axis,traintime);
plot(axis,traintimellm);
title('Train Time');
legend('proposed knn knn2','LLM');

figure;hold on;
plot(axis,testtimepro+traintime);
plot(axis,testtimeknn+traintime);
plot(axis,testtimeknn2+traintime);
plot(axis,testtimellm+traintimellm);
title('Total Time');
legend('proposed','knn','knn2','LLM');
%}
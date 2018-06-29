addpath('learner')

%% 2-dimension
times=1;
num_gradient=10000;
errorthreshold=0.1;
plotf=1;

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
%{
%ŽlŠp
tox=-5.3:0.2:5.3;
toy=7.5:0.1:10;
test_output=[tox(1,1)*ones(size(toy,2),1),toy';
            tox',toy(1,size(toy,2))*ones(size(tox,2),1);
            tox(1,size(tox,2))*ones(size(toy,2),1),flipud(toy');
            flipud(tox'),toy(1,1)*ones(size(tox,2),1)];
%}
%¯?
to1=9:0.2:12;
to2=11.9:-0.2:9.1;
to3=0:0.2:4.6;
to4=4.5:-0.2:0.1;
to5=0:-0.2:-4.6;
to6=-4.5:0.2:-0.1;
test_output=[zeros(size(to1,2),1),to1';
            zeros(size(to2,2),1),to2';
            to3',0.4*to3'+9*ones(size(to3,2),1);
            to4',0.4*to4'+9*ones(size(to4,2),1);
            to3',-0.5*to3'+9*ones(size(to3,2),1);
            to4',-0.5*to4'+9*ones(size(to4,2),1);
            to5',0.5*to5'+9*ones(size(to5,2),1);
            to6',0.5*to6'+9*ones(size(to6,2),1);
            to5',-0.4*to5'+9*ones(size(to5,2),1);
            to6',-0.4*to6'+9*ones(size(to6,2),1);];



% ‰Šú‰»
for q=1
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
end

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
    
    %% proposed
    tic
    [Cinputhat,COut,Cin]=TRR2(output,Cinput,test_output,number_of_redundant,k,k2);
    testtimepro(t,1)=toc;
    inputhat=cell(size(Cinputhat));
    
    % inveres f
    for i=1:size(Cinputhat,2)
        inputhat{i}(:,1)=Cinputhat{i}(:,3);
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
    inputhatUse=zeros(size(inputhat,2),2);
    outputhatUse=zeros(size(inputhat,2),2);
    for i=1:size(inputhat,2)
        signal=inputhat{i};
        outputhat=redundant_arm_dim2(signal,l1,l2);
        if i==1
            if size(signal,1)==1
                inputhatUse(1,:)=signal;
                outputhatUse(1,:)=outputhat;
            else
                [~,ind]=min(signal(:,1).^2+signal(:,2).^2);
                inputhatUse(1,:)=signal(ind,:);
                outputhatUse(1,:)=outputhat(ind,:);
            end
        else
            if size(signal,1)==1
                inputhatUse(i,:)=signal;
                outputhatUse(i,:)=outputhat;
            else
                [~,ind]=min((inputhatUse(i-1,1)-signal(:,1)).^2+(inputhatUse(i-1,2)-signal(:,2)).^2);
                inputhatUse(i,:)=signal(ind,:);
                outputhatUse(i,:)=outputhat(ind,:);
            end
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
    end
    
    if mod(t,plotf)==0
        figure;
        subplot(1,2,1)
        hold on;
        scatter(output(:,1),output(:,2),'.');
        plot(outputhatUse(:,1),outputhatUse(:,2),'LineWidth',2);
        plot(test_output(:,1),test_output(:,2))
        legend('data','estimate','true');
        title('Output space a proposed method');
        subplot(1,2,2)
        hold on;
        scatter(input(:,1),input(:,2),'.');
        plot(inputhatUse(:,1),inputhatUse(:,2),'LineWidth',2);  
        legend('data','estimate');
        title('Input space');
    end
    
    errorvar(t,1)=var(error);
    errorave(t,1)=sum(error)/size(error,1);
    gooderrorrate(t,1)=gooderror(t,1)/size(error,1);
    errorbar11(t,1)=max(error)-errorave(t,1);
    errorbar12(t,1)=errorave(t,1)-min(error);
                
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
        Input2(i,1)=IDW(database(indexes,1),database(indexes,2),input(indexes,1),test_output(i,1),test_output(i,2),-2,'ng',length(indexes));
        Input2(i,2)=IDW(database(indexes,1),database(indexes,2),input(indexes,2),test_output(i,1),test_output(i,2),-2,'ng',length(indexes));
        
        Output2(i,:)=redundant_arm_dim2(Input2(i,:),l1,l2);
        error3(i,1)=norm(Output2(i,:)-test_output(i,:));
        if error3(i,1)<errorthreshold
            gooderror2(t,1)=gooderror2(t,1)+1;
        end
    end
    if mod(t,plotf)==0
        figure;
        subplot(1,2,1)
        hold on;
        scatter(output(:,1),output(:,2),'.');
        plot(Output2(:,1),Output2(:,2),'LineWidth',2);
        plot(test_output(:,1),test_output(:,2))
        legend('data','estimate','true');
        title('Output space knn');
        subplot(1,2,2)
        hold on;
        scatter(input(:,1),input(:,2),'.');
        plot(Input2(:,1),Input2(:,2),'LineWidth',2);  
        legend('data','estimate');
        title('Input space');
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
    input4=zeros(size(test_output,1),2);
    Output4=zeros(size(test_output,1),2);
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
        input4(i,1)=IDW(Odatabase2(indexes3,1),Odatabase2(indexes3,2),kinput(indexes3,1),test_output(i,1),test_output(i,2),-2,'ng',k2);
        input4(i,2)=IDW(Odatabase2(indexes3,1),Odatabase2(indexes3,2),kinput(indexes3,2),test_output(i,1),test_output(i,2),-2,'ng',k2);
        
        Output4(i,:)=redundant_arm_dim2(input4(i,:),l1,l2);
        error4(i,1)=norm(Output4(i,:)-test_output(i,:));
        if error4(i,1)<errorthreshold
            gooderror3(t,1)=gooderror3(t,1)+1;
        end
    end
    if mod(t,plotf)==0
        figure;
        subplot(1,2,1)
        hold on;
        scatter(output(:,1),output(:,2),'.');
        plot(Output4(:,1),Output4(:,2),'LineWidth',2);
        plot(test_output(:,1),test_output(:,2))
        legend('data','estimate','true');
        title('Output space knn2');
        subplot(1,2,2)
        hold on;
        scatter(input(:,1),input(:,2),'.');
        plot(input4(:,1),input4(:,2),'LineWidth',2);  
        legend('data','estimate');
        title('Input space');
    end
    testtimeknn2(t,1)=toc;
    errorvarknn2(t,1)=var(error4);
    errorAveknn2(t,1)=sum(error4)/size(error4,1);
    gooderrorrate3(t,1)=gooderror3(t,1)/size(error4,1);
    errorbar31(t,1)=max(error4)-errorAveknn2(t,1);
    errorbar32(t,1)=errorAveknn2(t,1)-min(error4);     
    
 
end

axis=1*num_gradient:num_gradient:num_gradient*times;
figure;hold on;
plot(axis,errorave);
plot(axis,errorAveknn);
plot(axis,errorAveknn2);
title('errorAverage');
legend('proposed','knn','knn2');

figure;hold on;
plot(axis,gooderrorrate);
plot(axis,gooderrorrate2);
plot(axis,gooderrorrate3);
title('under threshold error percentage');
legend('proposed','knn','knn2');

figure;hold on;
errorbar(axis,gooderrorrate,errorbar12-errorbar11);
errorbar(axis,gooderrorrate2,errorbar22-errorbar21);
errorbar(axis,gooderrorrate3,errorbar32-errorbar31);
title('under threshold error percentage');
legend('proposed','knn','knn2');

figure;hold on;
errorbar(axis,errorave,errorbar12-errorbar11);
errorbar(axis,errorAveknn,errorbar22-errorbar21);
errorbar(axis,errorAveknn2,errorbar32-errorbar31);
title('errorAverage');
legend('proposed','knn','knn2');

figure;hold on;
plot(axis,errorvar);
plot(axis,errorvarknn);
plot(axis,errorvarknn2);
title('errorVariance');
legend('proposed','knn','knn2');

figure;hold on;
plot(axis,testtimepro+traintime);
plot(axis,testtimeknn+traintime);
plot(axis,testtimeknn2+traintime);
title('Total Time');
legend('proposed','knn','knn2');
%}
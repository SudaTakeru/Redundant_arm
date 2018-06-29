addpath('learner')

%% 2-dimension
times=100;
num_gradient=2000;
errorthreshold=0.1;
plotf=20;

Nt=20;
k=30;
k2=5;
k3=8;
grids=0;

sita1=[-pi/4 pi/4];
sita2=[-pi/2 pi/2];
sita3=[-pi/2 pi/2];
fai1=[0 0];

l1=10;
l2=5;
l3=3;
number_of_redundant=5;

%load('testinput_dim2input3')
% test point
tox=-5.3:0.2:5.3;
toy=11:0.1:13.5;
test_output=[tox(1,1)*ones(size(toy,2),1),toy';
            tox',toy(1,size(toy,2))*ones(size(tox,2),1);
            tox(1,size(tox,2))*ones(size(toy,2),1),flipud(toy');
            flipud(tox'),toy(1,1)*ones(size(tox,2),1)];

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
        end

input=[];
output=[];
Cinput=[];
IN=3;

for t=1:times

    N=num_gradient;

    tinput=zeros(N,3);
    tinput(:,1)=ones(N,1)*sita1(1,1)+(sita1(1,2)-sita1(1,1))*rand(N,1);
    tinput(:,2)=ones(N,1)*sita2(1,1)+(sita2(1,2)-sita2(1,1))*rand(N,1);
    tinput(:,3)=ones(N,1)*sita3(1,1)+(sita3(1,2)-sita3(1,1))*rand(N,1);
    toutput=redundant_arm_dim2_input3(tinput,l1,l2,l3);
    
    %% f| R^2 -> circle*R^1
    %{
    tCinput(:,1)=cos(tinput(:,2));
    tCinput(:,2)=sin(tinput(:,2));
    tCinput(:,3)=tinput(:,1);
    %}
    %% train
    tic
    input=[input;tinput];
    output=[output;toutput];
    %Cinput=[Cinput;tCinput];
    traintime(t,1)=toc;
    
    
    %% proposed
    tic
    %[inputhat,COut,Cin]=TRR2(output,input,test_output,number_of_redundant,k,k2);
    [inputhat,Results2,COut,Cin]=TRR5(output,input,test_output,number_of_redundant,k,k2,k3,ips);
    
    testtimepro(t,1)=toc;
    
    % show result
    
    error=[];
    errore=cell(size(inputhat,2),1);
    i2=1;
    inputhatUse=zeros(size(inputhat,2),3);
    outputhatUse=zeros(size(inputhat,2),2);
    for i=1:size(inputhat,2)
        signal=inputhat{i}{1};
        outputhat=redundant_arm_dim2(signal,l1,l2);
        if i==1
            if size(signal,1)==1
                inputhatUse(1,:)=signal;
                outputhatUse(1,:)=outputhat;
            else
                [~,ind]=min(signal(:,1).^2+signal(:,2).^2+signal(:,3).^2);
                inputhatUse(1,:)=signal(ind,:);
                outputhatUse(1,:)=outputhat(ind,:);
            end
        else
            if size(signal,1)==1
                inputhatUse(i,:)=signal;
                outputhatUse(i,:)=outputhat;
            else
                [~,ind]=min((inputhatUse(i-1,1)-signal(:,1)).^2+(inputhatUse(i-1,2)-signal(:,2)).^2+(inputhatUse(i-1,3)-signal(:,3)).^2);
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
        plot(outputhatUse(:,1),outputhatUse(:,2));
        plot(test_output(:,1),test_output(:,2))
        legend('data','estimate','true');
        title('Output space a proposed method');
        subplot(1,2,2)
        hold on;
        %scatter3(input(:,1),input(:,2),input(:,3),'.');
        plot3(inputhatUse(:,1),inputhatUse(:,2),inputhatUse(:,3));  
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
    Input2=zeros(size(test_output,1),IN);
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
        for dim=1:size(Input2,2)
            Input2(i,dim)=IDW2(database(indexes,:),input(indexes,dim),test_output(i,:),k2,-2);
        end
        Output2(i,:)=redundant_arm_dim2_input3(Input2(i,:),l1,l2,l3);
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
        plot(Output2(:,1),Output2(:,2));
        plot(test_output(:,1),test_output(:,2))
        legend('data','estimate','true');
        title('Output space knn');
        subplot(1,2,2)
        hold on;
        scatter3(input(:,1),input(:,2),input(:,3),'.');
        plot3(Input2(:,1),Input2(:,2),Input2(:,3));  
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
    input4=zeros(size(test_output,1),IN);
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
        for dim=1:size(input4,2)
            input4(i,dim)=IDW2(Odatabase2(indexes3,:),kinput(indexes3,dim),test_output(i,:),k2,-2);
        end
        Output4(i,:)=redundant_arm_dim2_input3(input4(i,:),l1,l2,l3);
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
        plot(Output4(:,1),Output4(:,2));
        plot(test_output(:,1),test_output(:,2))
        legend('data','estimate','true');
        title('Output space knn2');
        subplot(1,2,2)
        hold on;
        scatter3(input(:,1),input(:,2),input(:,3),'.');
        plot3(input4(:,1),input4(:,2),input4(:,3));  
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
%plot(axis,errorAve5);
title('errorAverage dim2 input3 ');
legend('proposed','knn','knn2','LLM');

figure;hold on;
plot(axis,gooderrorrate);
plot(axis,gooderrorrate2);
plot(axis,gooderrorrate3);
%plot(axis,gooderrorrate5);
title('under threshold error percentage  dim2 input3');
legend('proposed','knn','knn2','LLM');

figure;hold on;
errorbar(axis,gooderrorrate,errorbar12-errorbar11);
errorbar(axis,gooderrorrate2,errorbar22-errorbar21);
errorbar(axis,gooderrorrate3,errorbar32-errorbar31);
%errorbar(axis,gooderrorrate5,errorbar52-errorbar51);
title('under threshold error percentage  dim2 input3');
legend('proposed','knn','knn2','LLM');

figure;hold on;
errorbar(axis,errorave,errorbar12-errorbar11);
errorbar(axis,errorAveknn,errorbar22-errorbar21);
errorbar(axis,errorAveknn2,errorbar32-errorbar31);
%errorbar(axis,errorAve5,errorbar52-errorbar51);
title('errorAverage  dim2 input3');
legend('proposed','knn','knn2','LLM');

figure;hold on;
plot(axis,errorvar);
plot(axis,errorvarknn);
plot(axis,errorvarknn2);
%plot(axis,errorvar5);
title('errorVariance  dim2 input3');
legend('proposed','knn','knn2','LLM');

figure;
plot(axis,True_answerrate);
title('True rate of number of answer  dim2_input3');

figure;hold on;
plot(axis,testtimepro);
plot(axis,testtimeknn);
plot(axis,testtimeknn2);
%plot(axis,testtimellm);
title('Test Time  dim2 input3');
legend('proposed','knn','knn2','LLM');

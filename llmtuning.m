%% Type5 LLM
errorthreshold=0.1;
www=10;
Nt=20;
k=30;
k2=5;
grids=0;

sita1=[-pi/4 pi/4];
sita2=[-pi pi];
fai1=[0 0];

l1=10;
l2=3;
% errorAve5=cell(1,www);
% gooderror5=cell(1,www);
% gooderrorrate5=cell(1,www);
% Type5time=cell(1,www);
din = 2;
dout = 2;
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
N=5000;

input=[];
output=[];
Cinput=[];
tinput=zeros(N,2);
tinput(:,1)=ones(N,1)*sita1(1,1)+(sita1(1,2)-sita1(1,1))*rand(N,1);
tinput(:,2)=ones(N,1)*sita2(1,1)+(sita2(1,2)-sita2(1,1))*rand(N,1);
toutput=redundant_arm_dim2(tinput,l1,l2);
input=[input;tinput];
output=[output;toutput];

for t=1:www
    %% LLM
    error5=zeros(size(test_output,1),1);
    din = 2;
    dout = 2;
    lspecs = {'class', 'LLM'; 'lrate', 1.0; 'radius', 0.3; 'softmax',  0+0.4*t};
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
    gooderrorrate5(t,1)=gooderror5(t,1)/size(test_output,1);
    errorAve5(t,1)=sum(error5)/size(error5,1);
    errorbar51(t,1)=max(error5)-errorAve5(t,1);
    errorbar52(t,1)=errorAve5(t,1)-min(error5);
end

figure;
plot(1:www,gooderrorrate5)
%legend('1','2','3','4','5','6','7','8','9','10');
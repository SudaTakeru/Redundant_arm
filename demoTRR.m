



%% parameter 
k=30;
k2=5;
number_of_redundant=5;

N=1000;

%% redundant arm parameter
sita1=[-pi/4 pi/4];
sita2=[-pi pi];
fai1=[0 0];

l1=10;
l2=3;

intervals=10;
input1=sita1(1,1):(sita1(1,2)-sita1(1,1))/intervals:sita1(1,2);
input2=sita2(1,1):(sita2(1,2)-sita2(1,1))/intervals:sita2(1,2);
%%
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


input=zeros(N,2);
input(:,1)=ones(N,1)*sita1(1,1)+(sita1(1,2)-sita1(1,1))*rand(N,1);
input(:,2)=ones(N,1)*sita2(1,1)+(sita2(1,2)-sita2(1,1))*rand(N,1);
output=redundant_arm_dim2(input,l1,l2);


[Cinputhat,COut,Cin]=TRR2(output,input,test_output,number_of_redundant,k,k2);





    
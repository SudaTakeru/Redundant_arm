nn=8;

interval=3;

output=cell(1,nn);
input=cell(1,nn);
sita1=[-pi/4 pi/4];
sita2=[-pi/2 pi/2];
sita3=[-pi/2 pi/2];

for p=1:nn
    %% 2-dimension
    
    if p<5
        sita3=[-pi/2 0];
    else
        sita3=[0 pi/2];
    end
    
    if p==1 || p==5
        sita1=[-pi/4 0];
        sita2=[0  pi/2];
    elseif p==2 || p==6
        sita1=[0 pi/4];
        sita2=[0  pi/2];
        fai1=[0 0];
    elseif p==3 || p==7
        sita1=[-pi/4 0];
        sita2=[-pi/2 0];
        fai1=[0 0];        
    elseif p==4 || p==8
        sita1=[0 pi/4];
        sita2=[-pi/2 0];
    end
        
    l1=10;
    l2=5;
    l3=3;
    
    input1=sita1(1,1):(sita1(1,2)-sita1(1,1))/interval:sita1(1,2);
    input2=sita2(1,1):(sita2(1,2)-sita2(1,1))/interval:sita2(1,2);
    input3=sita3(1,1):(sita3(1,2)-sita3(1,1))/interval:sita3(1,2);

    c=1;
    input{p}=[];
    for i=1:size(input1,2)
        for ii=1:size(input2,2)
            for iii=1:size(input3,2)
                input{p}(c,:)=[input1(1,i),input2(1,ii),input3(1,iii)];
                c=c+1;
            end
        end
    end
    
    output{p}=redundant_arm_dim2_input3(input{p},l1,l2,l3);
    %{
    num=30;
    signal=output(50,:);
    
    nodes=output;
    indexes = zeros(num, 1);
    sqDists = zeros(num, 1);
    D = sum(((nodes - repmat(signal, size(nodes, 1), 1)).^2), 2);
    for i = 1:num
        [sqDists(i), indexes(i)] = min(D);
        D(indexes(i)) = inf;
    end
    %}
    
end
figure;hold on;
for p=1:nn
    if p>9
    scatter(output{p}(:,1),output{p}(:,2),'r')
    else
    scatter(output{p}(:,1),output{p}(:,2))
    end
end
figure;hold on;
for p=1:nn
    if p>9
    scatter3(input{p}(:,1),input{p}(:,2),input{p}(:,3),'r')
    else
    scatter3(input{p}(:,1),input{p}(:,2),input{p}(:,3))
    end
end
%{
figure;hold on;
Ninput=cell(1,nn);
for p=1:nn
    Ninput{p}(:,1)=cos(input{p}(:,2));
    Ninput{p}(:,2)=sin(input{p}(:,2));
    Ninput{p}(:,3)=input{p}(:,1);
    if p>4
        scatter3(Ninput{p}(:,1),Ninput{p}(:,2),Ninput{p}(:,3),'r')
    else
        scatter3(Ninput{p}(:,1),Ninput{p}(:,2),Ninput{p}(:,3))
    end
end

figure;hold on;
inputhat=cell(size(input));
for p=1:nn
    inputhat{p}(:,1)=Ninput{p}(:,3);
    for j=1:size(Ninput{p},1)
        if Ninput{p}(j,2)>0
            inputhat{p}(j,2)=acos(Ninput{p}(j,1));
        elseif Ninput{p}(j,2)<0
            inputhat{p}(j,2)=-1*acos(Ninput{p}(j,1));
        end
        
    end
    if p>4
        scatter(inputhat{p}(:,1),inputhat{p}(:,2),'r')
    else
        scatter(inputhat{p}(:,1),inputhat{p}(:,2))
    end
end
%}

figure;hold on;
out=[];
for ii=1:8
    out=[out;output{ii}];
    scatter(output{ii}(:,1),output{ii}(:,2),'b')
end
ind=cell(nn,size(output{1},1));
threshold=0.8;
for p=1:8
    out2=out;
    out2(1+(p-1)*size(output{ii},1):p*size(output{ii},1),:)=[];
    if 1
        for i=1:size(output{p},1)
            DD=out2-repmat(output{p}(i,:),size(out2,1),1);
            DD2=DD(:,1).^2+DD(:,2).^2;
            %DD2(i+(p-1)*size(output{p},1),1)=inf;
            [ind{p,i},~]=find(DD2<threshold);
        end
    end
    
    for i=1:size(output{p},1)
        scatter(out2(ind{p,i},1),out2(ind{p,i},2),'r')
    end
end


figure;hold on;
in=[];

for ii=1:8
    in=[in;input{ii}];
    scatter3(input{ii}(:,1),input{ii}(:,2),input{ii}(:,3),'b')
end
inind=zeros(size(in,1),1);
for p=1:8
    in2=in;
    in2(1+(p-1)*size(output{ii},1):p*size(output{ii},1),:)=[];
    
    for i=1:size(input{p},1)
        scatter3(in2(ind{p,i},1),in2(ind{p,i},2),in2(ind{p,i},3),'r')
        for j=1:size(in,1)
            for k=1:size(ind{p,i},1)
                if in(j,:)==in2(ind{p,i}(k,:),:)
                    inind(j,1)= inind(j,1)+1;
                end
            end
        end
    end
    
end

[ind0,~]=find(inind==0);
save('testinput_dim2input3','in','ind0')
%}


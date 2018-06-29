nn=6;

output=cell(1,nn);
input=cell(1,nn);
for p=1:nn
    %% 2-dimension
    interval=30;
    if p==1
        sita1=[-pi/4 0];
        sita2=[0  pi];
%         sita1=[-pi/4 0];
%         sita2=[0 pi];
        fai1=[0 0];
    elseif p==2
        sita1=[0 pi/4];
        sita2=[0  pi];
        fai1=[0 0];
    elseif p==3
        sita1=[-pi/4 0];
        sita2=[-pi 0];
        fai1=[0 0];        
    elseif p==4
        sita1=[0 pi/4];
        sita2=[-pi 0];
        fai1=[0 0];
    elseif p==5
        sita1=[-pi/4 pi/4];
        sita2=[-pi -pi];
        fai1=[0 0];
    else
        sita1=[-pi/4 pi/4];
        sita2=[pi pi];
        fai1=[0 0];
    end
        
    l1=10;
    l2=3;
    
    input1=sita1(1,1):(sita1(1,2)-sita1(1,1))/interval:sita1(1,2);
    if p<5
        input2=sita2(1,1):(sita2(1,2)-sita2(1,1))/interval:sita2(1,2);
    else
        input2=sita2(1,1);
    end
    c=1;
    input{p}=[];
    for i=1:size(input1,2)
        for ii=1:size(input2,2)
            input{p}(c,:)=[input1(1,i),input2(1,ii)];
            c=c+1;
        end
    end
    
    y=l1*cos(input{p}(:,1))+l2*cos(input{p}(:,1)+input{p}(:,2));
    x=l1*sin(input{p}(:,1))+l2*sin(input{p}(:,1)+input{p}(:,2));
    output{p}=[x,y];
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
    if p>4
    scatter(output{p}(:,1),output{p}(:,2),'r')
    else
    scatter(output{p}(:,1),output{p}(:,2))
    end
end
figure;hold on;
for p=1:nn
    if p>4
    scatter(input{p}(:,1),input{p}(:,2),'r')
    else
    scatter(input{p}(:,1),input{p}(:,2))
    end
end
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
for ii=1:4
    out=[out;output{ii}];
    scatter(output{ii}(:,1),output{ii}(:,2),'b')
end
ind=cell(nn,size(output{1},1));
threshold=0.1;
for p=1:4
    out2=out;
    out2(1+(p-1)*size(output{ii},1):p*size(output{ii},1),:)=[];
    if p==2 || p==3 || p==4 || p==1
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
for ii=1:4
    in=[in;inputhat{ii}];
    scatter(inputhat{ii}(:,1),inputhat{ii}(:,2),'b')
end

for p=1:4
    in2=in;
    in2(1+(p-1)*size(output{ii},1):p*size(output{ii},1),:)=[];
    
    for i=1:size(inputhat{p},1)
        scatter(in2(ind{p,i},1),in2(ind{p,i},2),'r')
    end
    
end
%}


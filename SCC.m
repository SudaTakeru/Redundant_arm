function ind=SCC(dataset,r)
% Simple Complex Clustering
% dataset = number of sample * dim
nu=size(dataset,1);
D=zeros(nu,nu);

for i=1:nu
    for j=1:nu
        if i<j
            D(i,j)=norm(sqrt((dataset(i,:)-dataset(j,:)).^2));            
        else
            D(i,j)=inf;
        end
    end
end

[DM,DM2]=find(D<r);
ind=zeros(nu,1);
ind0=zeros(nu,1);
for i=1:size(DM,1)
    if ind0(DM(i,1),1)==0 && ind0(DM2(i,1),1)==0 
        ind0(DM(i,1),1)= max(ind0)+1;
        ind0(DM2(i,1),1)= ind0(DM(i,1),1);
    elseif ind0(DM(i,1),1)==0 && ind0(DM2(i,1),1)~=0
        ind0(DM(i,1),1)=ind0(DM2(i,1),1);
    elseif ind0(DM(i,1),1)~=0 && ind0(DM2(i,1),1)==0
        ind0(DM2(i,1),1)= ind0(DM(i,1),1);
    elseif ind0(DM(i,1),1)~=0 && ind0(DM2(i,1),1)~=0
        if ind0(DM(i,1),1)~= ind0(DM2(i,1),1)
            [ti,~]=find(ind0==ind0(DM2(i,1),1));
            ind0(ti,1)=ind0(DM(i,1),1);
        end
    end
end

ind1=sort(ind0);

num=ind1(1,1);
for i=2:size(ind1,1)
    if ind1(i,1)~=ind1(i-1,1)
        num0=ind1(i,1);
        num=[num;num0];
    end    
end
if num(1,1)~=0
    for i=1:size(num,1)
        [ti2,~]=find(ind0==num(i,1));
        ind(ti2,1)=i;
    end
else
    [ti2,~]=find(ind0==0);
    for i=1:size(ti2,1)
        ind(ti2(i,1))=i;
    end
    iso=i;
    for i=2:size(num,1)
        [ti2,~]=find(ind0==num(i,1));
        ind(ti2,1)=i+iso-1;
    end
end

end
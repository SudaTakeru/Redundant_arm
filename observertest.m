clc; clear; close all;
import edu.stanford.math.plex4.*;

%addpath('function');

%% parameter
N=1000;
N=N/4;
Nt=10;
k=30;

k2=5;
Csize=5;
time=10;

c=5;

%% dataset cercle
x=-1+2*rand(N,1);
y=-1+2*rand(N,1);

y1=sqrt(1-x.^2);
y2=-1*sqrt(1-x.^2);
x1=sqrt(1-y.^2);
x2=-1*sqrt(1-y.^2);

data=[x,y1;x,y2;x1,y;x2,y];

%% test
testy=-1+2*rand(Nt,1);
testx=cell(1,Nt);
for j=1:size(testy,1)
    signal=testy(j,1);
    dataset=data(:,2);
    indexes = zeros(k, 1);
    sqDists = zeros(k, 1);
    D = sum(((dataset - repmat(signal, size(dataset, 1), 1)).^2), 2);
    for i = 1:k
        [sqDists(i), indexes(i)] = min(D);
        D(indexes(i)) = inf;
    end
    testx{j}=data(indexes,:);
end

%% Bayes
% P(x|y)を求めたい。
% 円だとP(y|x)もわからない
% P(x)=1/k
% P(y)=1/(N+1)

%% SOINN
%{
NN=cell(1,Nt);
clusternumber=cell(1,Nt);
for j=1:Nt
    trainingData=testx{j};
    [num,d]=size(trainingData);
    lambda=num*time+1;
    age=num*time+1;
    NN{j} =Soinn('lambda', lambda, 'ageMax', age, 'dim', d);
    % all landmarks learning
    for ii=1:time
    for i = 1:num
        NN{j}.inputSignal(trainingData(i, :));
        trainingData=trainingData+0.0001*rand(size(trainingData));
    end
    end
    UU=NN{j}.clustering(Csize);
    clusternumber{j}=max(UU);
    
end

% 不安定恣意的になってしまう。
% yでの連結成分を１、xでの連結成分を調べる方法はどうか？この場合Kは大きすぎない方が良い

vars=cell(Nt,1);
for i=1%:Nt
    %{
    figure;hold on;
    scatter(data(:,1),data(:,2))
    plot([1;-1],[testy(i,1);testy(i,1)]);
    scatter(testx{i}(:,1),testx{i}(:,2));
    scatter(NN{i}.nodes(:,1),NN{i}.nodes(:,2),'.');
    legend('data','y','neighborhood','nodes');
    %}
    vars{i}=var(NN{i}.nodes);
    wt=whitening(testx{i});
    %figure;
    %scatter(wt(:,1),wt(:,2));
end
%}
%% TDA
%{ 
1  そもそも近傍の連結数を数えればよいな
   近傍たちを白色化して0次persistent homology群を計算

2   K点で構成した連結数の違いを生むために複数個の円をデータセットとし
    そのpersistent homology群を計算
    データセットの作り方の工夫が大事？

3   近傍のpersistent homology群がどのデータセットに近いかによって
    そのテスト点の冗長差を表現できる。

%}
intervals=cell(1,Nt);
Sintervals=cell(1,c);
bars=cell(1,Nt);
Sbars=cell(1,c);

min_dimension=0;
max_dimension = 2;
division_time=20;

% 1
for j=1:Nt
    nodes=whitening(testx{j});
    
    R=max(max(nodes*nodes'));
    max_filtration_value = R;
    
    num_divisions =division_time*size(nodes,1);
    point_cloud2=nodes;
    random_selector = api.Plex4.createRandomSelector(point_cloud2, size(nodes,1));
    %stream = api.Plex4.createWitnessStream(random_selector, max_dimension, max_filtration_value, num_divisions);
    stream = api.Plex4.createVietorisRipsStream(random_selector, max_dimension, max_filtration_value, num_divisions);
    
    % print out the size of the stream
    %num_simplices = stream.getSize();
    
    % get persistence algorithm over Z/2Z
    persistence = api.Plex4.getModularSimplicialAlgorithm(max_dimension, 2);
    
    % compute the intervals
    intervals{j}= persistence.computeIntervals(stream);
    %persistence_diagram(intervals{j}, min_dimension, max_dimension);
    % create the barcode plots
    
    options.filename = 'rips';
    options.max_filtration_value = max_filtration_value;
    options.max_dimension = max_dimension - 1;
    %plot_barcodes(intervals{j}, options);
    bars{j}=barcodes(intervals{j}, options);
end

% 2
for j=1:c
    
    numc=round(k/j);
    sampledata=[];
    for i=1:j
        xi=2*(i-1);
        yi=2*(i-1);
        
        r=0.05+0.05*rand(numc,1);
        sita=0.5*pi*rand(numc,1);
        
        x=r.*cos(sita)+xi.*ones(numc,1);
        y=r.*sin(sita)+yi.*ones(numc,1);

        Sdata=[x,y];
        sampledata=[sampledata;Sdata];
    end
    
    nodes=whitening(sampledata);
    
    %figure;
    %scatter(nodes(:,1),nodes(:,2));
    
    R=max(max(nodes*nodes'));
    max_filtration_value = R;
    
    num_divisions =division_time*size(nodes,1);
    point_cloud2=nodes;
    random_selector = api.Plex4.createRandomSelector(point_cloud2, size(nodes,1));
    %stream = api.Plex4.createWitnessStream(random_selector, max_dimension, max_filtration_value, num_divisions);
    stream = api.Plex4.createVietorisRipsStream(random_selector, max_dimension, max_filtration_value, num_divisions);
    
    % print out the size of the stream
    %num_simplices = stream.getSize();
    
    % get persistence algorithm over Z/2Z
    persistence = api.Plex4.getModularSimplicialAlgorithm(max_dimension, 2);
    
    % compute the intervals
    Sintervals{j}= persistence.computeIntervals(stream);
    %persistence_diagram(intervals{j}, min_dimension, max_dimension);
    % create the barcode plots
    
    options.filename = 'rips';
    options.max_filtration_value = max_filtration_value;
    options.max_dimension = max_dimension - 1;
    %plot_barcodes(Sintervals{j}, options);
    Sbars{j}=barcodes(Sintervals{j}, options);
end

% 3
for iii=1
tic
nu=Nt;
intervals_dim0=cell(1,nu);
intervals_dim1=cell(1,nu);
intervals_dim2=cell(1,nu);
Sintervals_dim0=cell(1,c);
Sintervals_dim1=cell(1,c);
for j=1:nu
intervals_dim0{j}=intervals{j}.getIntervalsAtDimension(0);
intervals_dim1{j}=intervals{j}.getIntervalsAtDimension(1);
%intervals_dim2{j}=intervals{j}.getIntervalsAtDimension(2);
end
for j=1:c
Sintervals_dim0{j}=Sintervals{j}.getIntervalsAtDimension(0);
Sintervals_dim1{j}=Sintervals{j}.getIntervalsAtDimension(1);
end

bottleneck_distance_dim0=zeros(nu,c);
bottleneck_distance_dim1=zeros(nu,c);

for j=1:Nt
    for jj=1:c
        if bars{j}.num_intervals(1,1)~=0 && bars{jj}.num_intervals(1,1)~=0
            bottleneck_distance_dim0(j,jj) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(intervals_dim0{j},Sintervals_dim0{jj});
        else
            bottleneck_distance_dim0(j,jj) = 0;
        end
        if bars{j}.num_intervals(2,1)~=0 && bars{jj}.num_intervals(2,1)~=0
            bottleneck_distance_dim1(j,jj) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(intervals_dim1{j},Sintervals_dim1{jj});
        else
            bottleneck_distance_dim1(j,jj) = 0;
        end

    end
end
fprintf('calculate distance Time\n');
toc

numcluster=zeros(Nt,1);
for j=1:Nt
    [~,ind]=min(bottleneck_distance_dim0(j,:));
    numcluster(j,1)=ind;
end
end

% 4

vars=cell(Nt,1);
for j=1:Nt 
    
    figure;hold on;
    scatter(data(:,1),data(:,2))
    plot([1;-1],[testy(j,1);testy(j,1)]);
    scatter(testx{j}(:,1),testx{j}(:,2));
    %scatter(NN{i}.nodes(:,1),NN{i}.nodes(:,2),'.');
    legend('data','y','neighborhood','nodes');
    %}
    [idx,C]=kmeans(testx{j},numcluster(j,1));  % kmeansはダメ?連結をあらわせない?
    gscatter(testx{j}(:,1),testx{j}(:,2),idx);
    
    clusters=cell(1,numcluster(j,1));
    for ii=1:numcluster(j,1)
        for jj=1:size(idx,1)
            if idx(jj,1)==ii
                clusters{ii}=[clusters{ii};testx{j}(jj,:)];
            end
        end
    end
    outputx=zeros(numcluster(j,1),1);
    for ii=1:numcluster(j,1)
        outputx(ii,1)=IDW(clusters{ii}(:,1),clusters{ii}(:,1),clusters{ii}(:,2),testy(j,1),testy(j,1),-2,'ng',k2);
    end
    tx1=sqrt(1-testy(j,1).^2);
    tx2=-1*sqrt(1-testy(j,1).^2);
    if numcluster(j,1)==2
        error1=(min([tx1;tx2])-min(outputx))^2;
        error2=(max([tx1;tx2])-max(outputx))^2;
        error(1,j)=error1+error2;
    else
        error(1,j)=NaN;
    end
end

%% 比較対象
for ii=1:Nt
    outputx(ii,1)=IDW(data(:,2),data(:,2),data(:,1),testy(ii,1),testy(ii,1),-2,'ng',k2);
    tx1=sqrt(1-testy(ii,1).^2);
    tx2=-1*sqrt(1-testy(ii,1).^2);

    error1=(min([tx1;tx2])-min(outputx))^2;
    error2=(max([tx1;tx2])-max(outputx))^2;
    erroridw(1,ii)=error1+error2;

end

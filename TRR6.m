function [Results,Results2,clusters,clustersb]=TRR6(dataseta,datasetb,test,number_of_redundant,k,k2,k3,ips,r)
% topological redundant regresion
% dataseta is in compare space
% datasetb is in regression space
% test is in compare space
% Results is in regression space
% clusters is in compaare space
% k is k nearest neighorhoods in compare space
% k2 is k2 nearest neighborhood in regression space
% TRR use kmeans but TRR2 don't use kmeans. TRR2 use tda clustering.
% point or not
% 1st KNN --> r nn

import edu.stanford.math.plex4.*;

Nt=size(test,1);

%% First r neighborhood
testx=cell(1,Nt);
testb=cell(1,Nt);
for j=1:Nt
    signal=test(j,:);
    
    D = sum(((dataseta - repmat(signal, size(dataseta, 1), 1)).^2), 2);
    
    [indexes,~]=find(D(:,1)<r);
    if size(indexes,1)>2*k || size(indexes,1)<2
        indexes = zeros(k, 1);
        sqDists = zeros(k, 1);
        for i = 1:k
            [sqDists(i), indexes(i)] = min(D);
            D(indexes(i)) = inf;
        end
    end
    
    testx{j}=dataseta(indexes,:);
    testb{j}=datasetb(indexes,:);
end

%% TDA if the answer shape is some points
intervals=cell(1,Nt);
Sintervals=cell(1,number_of_redundant);
bars=cell(1,Nt);
Sbars=cell(1,number_of_redundant);

max_dimension = size(dataseta,2);%+size(datasetb,2);
division_time=20;

% 1 compute persistent homology for neighborhoods
for j=1:Nt
    %nodes=whitening(testb{j});
    nodes=zscore(testb{j});
    %nodes=testb{j};
    
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
    %figure;
    %scatter(nodes(:,1),nodes(:,2));
end

% 2 preparation sample persistnt homology
for j=1:number_of_redundant
    
    numc=round(k/j);
    sampledata=[];
    for i=1:j
        xi=2*(i-1);
        yi=2*(i-1);
        
        r=0.05+0.05*rand(numc,1);
        sita=2*pi*rand(numc,1);
        
        x=r.*cos(sita)+xi.*ones(numc,1);
        y=r.*sin(sita)+yi.*ones(numc,1);
        
        Sdata=[x,y];
        sampledata=[sampledata;Sdata];
    end
    
    %nodes=whitening(sampledata);
    nodes=zscore(sampledata);
    
    
    %figure;
    %scatter(nodes(:,1),nodes(:,2));
    
    R=max(nodes)-min(nodes);
    max_filtration_value = sqrt(R*R');
    
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

% 3 compute distance from sample to data
for iii=1
    
    nu=Nt;
    intervals_dim0=cell(1,nu);
    intervals_dim1=cell(1,nu);
    %intervals_dim2=cell(1,nu);
    Sintervals_dim0=cell(1,number_of_redundant);
    Sintervals_dim1=cell(1,number_of_redundant);
    for j=1:nu
        intervals_dim0{j}=intervals{j}.getIntervalsAtDimension(0);
        intervals_dim1{j}=intervals{j}.getIntervalsAtDimension(1);
        %intervals_dim2{j}=intervals{j}.getIntervalsAtDimension(2);
    end
    for j=1:number_of_redundant
        Sintervals_dim0{j}=Sintervals{j}.getIntervalsAtDimension(0);
        Sintervals_dim1{j}=Sintervals{j}.getIntervalsAtDimension(1);
    end
    
    bottleneck_distance_dim0=zeros(nu,number_of_redundant);
    bottleneck_distance_dim1=zeros(nu,number_of_redundant);
    
    for j=1:Nt
        for jj=1:number_of_redundant
            if bars{j}.num_intervals(1,1)~=0 && Sbars{jj}.num_intervals(1,1)~=0
                bottleneck_distance_dim0(j,jj) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(intervals_dim0{j},Sintervals_dim0{jj});
            else
                bottleneck_distance_dim0(j,jj) = 0;
            end
            if bars{j}.num_intervals(2,1)~=0 && Sbars{jj}.num_intervals(2,1)~=0
                bottleneck_distance_dim1(j,jj) = edu.stanford.math.plex4.bottleneck.BottleneckDistance.computeBottleneckDistance(intervals_dim1{j},Sintervals_dim1{jj});
            else
                bottleneck_distance_dim1(j,jj) = 0;
            end
        end
    end
    
    numcluster=zeros(Nt,1);
    for j=1:Nt
        [~,ind]=min(bottleneck_distance_dim0(j,:));
        numcluster(j,1)=ind;
    end
end

% 4 regression
Results=cell(1,Nt);
Results2=zeros(Nt,1);
clusters=cell(1,Nt);
clustersb=cell(1,Nt);
for j=1:Nt
    bari=bars{j}.endpoints{1}(:,2);
    r0=zeros(numcluster(j,1),1);
    for ii=1:numcluster(j,1)+1
        [~,Mind]=max(bari);
        r0(ii,1)=bari(Mind,1);
        bari(Mind,1)=0;
    end
    r=(r0(numcluster(j,1),1)+r0(numcluster(j,1)+1,1))/2;%Bar length
    idx=SCC(zscore(testb{j}),r);
    
    
    clusters{j}=cell(1,numcluster(j,1));
    clustersb{j}=cell(1,numcluster(j,1));
    output2=cell(1,numcluster(j,1));
    
    for ii=1:numcluster(j,1)
        for jj=1:size(idx,1)
            if idx(jj,1)==ii
                clusters{j}{ii}=[clusters{j}{ii};testx{j}(jj,:)];
                clustersb{j}{ii}=[clustersb{j}{ii};testb{j}(jj,:)];
            end
        end
        if size(clusters{j}{ii}, 1)>=k2
            k4=k2;
        else
            k4=size(clusters{j}{ii}, 1);
        end
        % point or other?
        count=0;
        gp=sum(clustersb{j}{ii})/size(clustersb{j}{ii},1);
        dgp=clustersb{j}{ii}-repmat(gp,size(clustersb{j}{ii},1),1);
        for jj=1:size(clustersb{j}{ii},1)
            if norm(dgp(jj,:))>ips
                count=count+1;
            end
        end
        if count>0
            Results2(j,1)=1;
            indexes = zeros(k3, 1);
            sqDists = zeros(k3, 1);
            D = sum(((clusters{j}{ii} - repmat(test(j,:), size(clusters{j}{ii}, 1), 1)).^2), 2);
            for l = 1:k3
                [sqDists(l), indexes(l)] = min(D);
                D(indexes(l)) = inf;
            end
            
            output=clustersb{j}{ii}(indexes,:);
        else
            Results2(j,1)=0;
            indexes = zeros(k4, 1);
            sqDists = zeros(k4, 1);
            D = sum(((clusters{j}{ii} - repmat(test(j,:), size(clusters{j}{ii}, 1), 1)).^2), 2);
            for l = 1:k4
                [sqDists(l), indexes(l)] = min(D);
                D(indexes(l)) = inf;
            end
            output=zeros(1,size(testb{j},2));
            for dim=1:size(testb{j},2)
                output(1,dim)=IDW2(clusters{j}{ii}(indexes,:),clustersb{j}{ii}(indexes,dim),test(j,:),length(indexes),-2);
            end
        end
        output2{ii}=output;
    end
    Results{j}=output2;
end
end
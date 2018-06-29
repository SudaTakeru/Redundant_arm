function [Results,clusters,clustersb]=TRR(dataseta,datasetb,test,number_of_redundant,k,k2)
% topological redundant regresion
% dataseta is in compare space
% datasetb is in regression space
% test is in compare space
% Results is in regression space
% clusters is in compaare space
% k is k nearest neighorhoods in compare space
% k2 is k2 nearest neighborhood in regression space

import edu.stanford.math.plex4.*;

Nt=size(test,1);

%% First K neareset neighborhood
testx=cell(1,Nt);
testb=cell(1,Nt);
for j=1:Nt
    signal=test(j,:);
    indexes = zeros(k, 1);
    sqDists = zeros(k, 1);
    D = sum(((dataseta - repmat(signal, size(dataseta, 1), 1)).^2), 2);
    for i = 1:k
        [sqDists(i), indexes(i)] = min(D);
        D(indexes(i)) = inf;
    end
    testx{j}=dataseta(indexes,:);
    testb{j}=datasetb(indexes,:);
end

%% TDA
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

% 3 compute didtance from sample to data
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
clusters=cell(1,Nt);
clustersb=cell(1,Nt);
for j=1:Nt
    
    [idx,~]=kmeans(testb{j},numcluster(j,1));
    clusters{j}=cell(1,numcluster(j,1));
    clustersb{j}=cell(1,numcluster(j,1));
    output=zeros(numcluster(j,1),size(testb{j},2));
    
    for ii=1:numcluster(j,1)
        for jj=1:size(idx,1)
            if idx(jj,1)==ii
                clusters{j}{ii}=[clusters{j}{ii};testx{j}(jj,:)];
                clustersb{j}{ii}=[clustersb{j}{ii};testb{j}(jj,:)];
            end
        end
        
        indexes = zeros(k2, 1);
        sqDists = zeros(k2, 1);
        D = sum(((clusters{j}{ii} - repmat(test(j,:), size(clusters{j}{ii}, 1), 1)).^2), 2);
        for l = 1:k2
            [sqDists(l), indexes(l)] = min(D);
            D(indexes(l)) = inf;
        end
        for dim=1:size(testb{j},2)
            output(ii,dim)=IDW2(clusters{j}{ii}(indexes,:),clustersb{j}{ii}(indexes,dim),test(j,:),length(indexes),-2);
        end
    end
    Results{j}=output;
end

end



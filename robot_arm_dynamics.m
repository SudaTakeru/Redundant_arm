
dim=2;

if dim==2
    for pppp=1
    %% 2-dimension
    interval=0.1;
    
    sita1=[-pi/4 pi/4];
    sita2=[0  pi];%[-pi/2 0];
    fai1=[0 0];
    
    l1=10;
    l2=3;
    
    input1=sita1(1,1):interval:sita1(1,2);
    input2=sita2(1,1):interval:sita2(1,2);
    
    c=1;
    input=[];
    for i=1:size(input1,2)
        for ii=1:size(input2,2)
            input(c,:)=[input1(1,i),input2(1,ii)];
            c=c+1;
        end
    end
    
    y=l1*cos(input(:,1))+l2*cos(input(:,1)+input(:,2));
    x=l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2));
    output=[x,y];
    
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
    figure;hold on;
    scatter(output(:,1),output(:,2))
    scatter(signal(1,1),signal(1,2),'.')
    scatter(output(indexes,1),output(indexes,2))
    
    figure;hold on;
    scatter(input(:,1),input(:,2))
    scatter(input(indexes,1),input(indexes,2))
    %}
    end
elseif dim==3
    %% 3-dimension
    for pppp=1
    done=0;
    testpointind=0;
    testpoint=[12,3,3];
    plotinterval=100;
    
    interval=0.05;
    
    sita1=[-pi/4 pi/4];
    sita2=[-pi pi];
    sita3=[-pi pi];
    
    l1=10;
    l2=3;
    
    input1=sita1(1,1):interval:sita1(1,2);
    input2=sita2(1,1):interval:sita2(1,2);
    input3=sita3(1,1):interval:sita3(1,2);
    if done==0
        c=1;
        input=[];
        for i=1:size(input1,2)
            for ii=1:size(input2,2)
                for iii=1:size(input3,2)
                    input(c,:)=[input1(1,i),input2(1,ii),input3(1,iii)];
                    c=c+1;
                end
            end
        end
        
        z=(l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2))).*cos(input(:,3));
        x=(l1*cos(input(:,1))+l2*cos(input(:,1)+input(:,2)));
        y=(l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2))).*sin(input(:,3));
        
        
        output=[x,y,z];
    end
    
    
    num=30;
    %signal=output(45524,:);
    %signal=output(44542,:);
    if testpointind>0
        signal=output(testpointind,:);
    else
        signal=testpoint;
    end
    nodes=output;
    indexes = zeros(num, 1);
    sqDists = zeros(num, 1);
    D = sum(((nodes - repmat(signal, size(nodes, 1), 1)).^2), 2);
    for i = 1:num
        [sqDists(i), indexes(i)] = min(D);
        D(indexes(i)) = inf;
    end
    
    ff=1:plotinterval:size(output,1);
    figure;hold on;grid on;
    scatter3(output(ff,1),output(ff,2),output(ff,3),'.')
    scatter3(signal(1,1),signal(1,2),signal(1,3))
    
    scatter3(output(indexes,1),output(indexes,2),output(indexes,3))
    xlabel('x');
    ylabel('y');
    zlabel('z');
    
    figure;hold on;grid on;
    scatter3(input(:,1),input(:,2),input(:,3),'.')
    scatter3(input(indexes,1),input(indexes,2),input(indexes,3))
    xlabel('sita1');
    ylabel('sita2');
    zlabel('sita3');

    %}
    end
end
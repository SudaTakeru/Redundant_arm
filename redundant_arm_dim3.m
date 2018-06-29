function output=redundant_arm_dim3(input,l1,l2)
z=(l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2))).*cos(input(:,3));
x=(l1*cos(input(:,1))+l2*cos(input(:,1)+input(:,2)));
y=(l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2))).*sin(input(:,3));
output=[x,y,z];
end
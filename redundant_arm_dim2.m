function output=redundant_arm_dim2(input,l1,l2)

y=l1*cos(input(:,1))+l2*cos(input(:,1)+input(:,2));
x=l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2));
output=[x,y];
end
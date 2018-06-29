function output=redundant_arm_dim2_input3(input,l1,l2,l3)

y=l1*cos(input(:,1))+l2*cos(input(:,1)+input(:,2))+l3*cos(input(:,1)+input(:,2)+input(:,3));
x=l1*sin(input(:,1))+l2*sin(input(:,1)+input(:,2))+l3*sin(input(:,1)+input(:,2)+input(:,3));
output=[x,y];
end
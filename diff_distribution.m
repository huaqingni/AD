load('ASD.mat')
load('ASD_typ.mat');
load('brain_region_4');
mode=cell(1,3);
mode{1}=reho;
mode{2}=alff;
mode{3}=vbm;
t_diff=[];
for i=1:3
    D=[];
    data=mode{i};
    data=data(:,1:90);
    data=normalize(data,2);
    
    
    
    A1=data(Y==1,:);
    A2=data(Y==2,:);
    A3=data(Y==3,:);
    A4=data(Y==4,:);
    
    u1=mean(A1,1);
    v1=std(A1,0,1);  
    dis1=u1./v1;
    u2=mean(A2,1);
    v2=std(A2,0,1);
    dis2=u2./v2;
    u3=mean(A3,1);
    v3=std(A3,0,1);
    dis3=u3./v3;
    u4=mean(A4,1);
    v4=std(A4,0,1);
    dis4=u4./v4;
    dis=[dis1;dis2;dis3;dis4];
    diff=[];
    for i=1:90     
        sum_diff=compute(dis(:,i));
         diff=[diff;sum_diff];
    end
    t_diff=[t_diff;diff'];
end
t_diff=sum(t_diff,1);
t_diff=t_diff(s);


function sum_diff = compute(data)
n = length(data);
sum_diff = 0;
for i = 1:n-1
    for j = i+1:n
        diff = abs(data(i) - data(j));
        sum_diff = sum_diff + diff;
    end
end
end






load('ASD.mat')
load('ASD_typ.mat');
load('brain_region_2');
mode=cell(1,3);
mode{1}=reho;
mode{2}=alff;
mode{3}=vbm;
D_t=[];
for i=1:3
    D=[];
    data=mode{i};
for i=s
    d=clu(data(:,i),Y);
    D=[D;d];
    
end
D=D'
D_t=[D_t;D];
end
function D=clu(x,Y)
a_x=mean(x);
a_x1=mean(x(Y~=0));
a_x0=mean(x(Y==0));
intra=sum((x(Y~=0)-a_x1).^2)+sum((x(Y==0)-a_x0).^2);
inter=sum((x(Y~=0)-a_x).^2)+sum((x(Y==0)-a_x).^2);
D=intra/inter;
return 
end








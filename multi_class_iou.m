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
    diff=[];
     for i=1:90 
        [f_x1,x1]=gassui(data(Y==1,i));
        [f_x2,x2]=gassui(data(Y==2,i));     
        [f_x3,x3]=gassui(data(Y==3,i));
        [f_x4,x4]=gassui(data(Y==4,i)); 
        f_x=[f_x1;f_x2;f_x3;f_x4];
        x=[x1;x2;x3;x4];
        sum_diff = compute(f_x,x);
        diff=[diff,sum_diff];
     end
 t_diff=[t_diff;diff];
end
t_diff=t_diff(:,s);
t_diff=sum(t_diff,1)/3;



function iou=IOU(f_x,x,f_y,y)
intersection_area = integral(@(z) min(interp1(f_x, x, z), interp1(f_y, y, z)), min(min(f_x), min(f_y)), max(max(f_x), max(f_y)));
union_area = integral(@(z) max(interp1(f_x, x, z), interp1(f_y, y, z)), min(min(f_x), min(f_y)), max(max(f_x), max(f_y)));
iou = intersection_area / union_area;
return
end

function  [f_x,f]=gassui(data)
   [f,f_x]=ksdensity(data);
    return
end

function sum_diff = compute(f_x,x)
s= size(f_x);
n=s(1);
sum_diff = 0;
for i = 1:n-1
    for j = i+1:n
        diff =IOU(f_x(i,:),x(i,:),f_x(j,:),x(j,:));
        sum_diff = sum_diff + diff;
    end
end
 return
end






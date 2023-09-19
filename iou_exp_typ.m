load('ASD.mat')
load('ASD_typ.mat');
load('brain_region_4');
brain_area={'中央前回(L)','中央前回(R)','背外侧额上回(L)','背外侧额上回(R)','眶部额上回(L)','眶部额上回(R)','额中回(L)','额中回(R)','眶部额中回(L)','眶部额中回(R)','岛盖部额下回(L)','岛盖部额下回(R)','三角部额下回(L)','三角部额下回(R)','眶部额下回(L)','眶部额下回(R)','中央沟盖(L)','中央沟盖(R)','补充运动区(L)','补充运动区(R)','嗅皮质(L)','嗅皮质(R)','内侧额上回(L)','内侧额上回(R)','眶内额上回(L)','眶内额上回(R)','回直肌(L)','回直肌(R)','脑岛(L)','脑岛(R)','前扣带和旁扣带脑回(L)','前扣带和旁扣带脑回(R)','内侧和旁扣带脑回(L)','内侧和旁扣带脑回(R)','后扣带回(L)','后扣带回(R)','海马(L)','海马(R)','海马旁回(L)','海马旁回(R)','杏仁核(L)','杏仁核(R)','距状裂周围皮层(L)','距状裂周围皮层(R)','楔叶(L)','楔叶(R)','舌回(L)','舌回(R)','枕上回(L)','枕上回(R)','枕中回(L)','枕中回(R)','枕下回(L)','枕下回(R)','梭状回(L)','梭状回(R)','中央后回(L)','中央后回(R)','顶上回(L)','顶上回(R)','顶下缘角回(L)','顶下缘角回(R)','缘上回(L)','缘上回(R)','角回(L)','角回(R)','楔前叶(L)','楔前叶(R)','中央旁小叶(L)','中央旁小叶(R)','尾状核(L)','尾状核(R)','豆状壳核(L)','豆状壳核(R)','豆状苍白球(L)','豆状苍白球(R)','丘脑(L)','丘脑(R)','颞横回(L)','颞横回(R)','颞上回(L)','颞上回(R)','颞极：颞上回(L)','颞极：颞上回(R)','颞中回(L)','颞中回(R)','颞极：颞中回(L)','颞极：颞中回(R)','颞下回(L)','颞下回(R)'};
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
         class=3;
        [f_x1,x1]=gassui(data(Y==class,i));
        [f_x2,x2]=gassui(data((Y~=class)&(Y~=0),i));
        f_x=[f_x1;f_x2];
        x=[x1;x2];
        sum_diff = compute(f_x,x);
        diff=[diff,sum_diff];
     end
 t_diff=[t_diff;diff];
end
t_diff=t_diff(:,s);
ave_diff=sum(t_diff,1)/3;
matrix=[t_diff;ave_diff];
numRows = size(matrix, 1);
minIndices = zeros(numRows, 5);  % 用于存储每行最小值的位置
brain_39=brain_area(s);
for i = 1:numRows
    row = matrix(i, :);
    [~, sortedIndices] = sort(row);
    minIndices(i, :) = sortedIndices(1:5);
end
total={};

for j=1:4
b_name=brain_39(minIndices(j,:));
iou=num2cell(matrix(j,minIndices(j,:)));
str=[b_name;iou];
total=[total;str];

end



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






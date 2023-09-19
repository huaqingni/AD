load('ASD.mat')
load('ASD_typ.mat');
load('brain_region_2');
brain_area={'中央前回(L)','中央前回(R)','背外侧额上回(L)','背外侧额上回(R)','眶部额上回(L)','眶部额上回(R)','额中回(L)','额中回(R)','眶部额中回(L)','眶部额中回(R)','岛盖部额下回(L)','岛盖部额下回(R)','三角部额下回(L)','三角部额下回(R)','眶部额下回(L)','眶部额下回(R)','中央沟盖(L)','中央沟盖(R)','补充运动区(L)','补充运动区(R)','嗅皮质(L)','嗅皮质(R)','内侧额上回(L)','内侧额上回(R)','眶内额上回(L)','眶内额上回(R)','回直肌(L)','回直肌(R)','脑岛(L)','脑岛(R)','前扣带和旁扣带脑回(L)','前扣带和旁扣带脑回(R)','内侧和旁扣带脑回(L)','内侧和旁扣带脑回(R)','后扣带回(L)','后扣带回(R)','海马(L)','海马(R)','海马旁回(L)','海马旁回(R)','杏仁核(L)','杏仁核(R)','距状裂周围皮层(L)','距状裂周围皮层(R)','楔叶(L)','楔叶(R)','舌回(L)','舌回(R)','枕上回(L)','枕上回(R)','枕中回(L)','枕中回(R)','枕下回(L)','枕下回(R)','梭状回(L)','梭状回(R)','中央后回(L)','中央后回(R)','顶上回(L)','顶上回(R)','顶下缘角回(L)','顶下缘角回(R)','缘上回(L)','缘上回(R)','角回(L)','角回(R)','楔前叶(L)','楔前叶(R)','中央旁小叶(L)','中央旁小叶(R)','尾状核(L)','尾状核(R)','豆状壳核(L)','豆状壳核(R)','豆状苍白球(L)','豆状苍白球(R)','丘脑(L)','丘脑(R)','颞横回(L)','颞横回(R)','颞上回(L)','颞上回(R)','颞极：颞上回(L)','颞极：颞上回(R)','颞中回(L)','颞中回(R)','颞极：颞中回(L)','颞极：颞中回(R)','颞下回(L)','颞下回(R)'};
mode=cell(1,3);
mode{1}=reho;
mode{2}=alff;
mode{3}=vbm;
mode_data=cell(1,90);
t_diff=[];
for j=1:90
    s_data=[];
for i=1:3
a=mode{i};
a=a(:,j);
s_data=[s_data,a];
end
mode_data{j}=s_data;
end
diff=[];
L=[];
for i=1:90 
    data=mode_data{i};
         class=4;
        g1=data(Y~=0,:);
        g2=data((Y~=0&Y~=class),:);
        g2=data((Y==0),:);
        [p_value,lamda] = computeWilksPValue(g1, g2);
        diff=[diff,p_value];
        L=[L,lamda];
end
 t_diff=diff;

t_diff=t_diff(:,s);
t_L=L(:,s);
roh=corr(t_diff',t_L');
matrix=t_diff;
numRows = size(matrix, 1);
minIndices = zeros(numRows, 5);  % 用于存储每行最小值的位置
brain_39=brain_area(s);
for i = 1:numRows
    row = matrix(i, :);
    [~, sortedIndices] = sort(row);
    minIndices(i, :) = sortedIndices(1:5);
end
min=s(minIndices);
total={};


for j=1:1
b_name=brain_39(minIndices(j,:));
iou=num2cell(matrix(j,minIndices(j,:)));
str=[b_name;iou];
total=[total;str];
end






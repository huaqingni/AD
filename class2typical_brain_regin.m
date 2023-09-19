listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe
index=1;

load('regionlxf.mat')

% Load the data and select features for classification
load(['ADS_all_region.mat'])

brain_area={'中央前回(L)','中央前回(R)','背外侧额上回(L)','背外侧额上回(R)','眶部额上回(L)','眶部额上回(R)','额中回(L)','额中回(R)','眶部额中回(L)','眶部额中回(R)','岛盖部额下回(L)','岛盖部额下回(R)','三角部额下回(L)','三角部额下回(R)','眶部额下回(L)','眶部额下回(R)','中央沟盖(L)','中央沟盖(R)','补充运动区(L)','补充运动区(R)','嗅皮质(L)','嗅皮质(R)','内侧额上回(L)','内侧额上回(R)','眶内额上回(L)','眶内额上回(R)','回直肌(L)','回直肌(R)','脑岛(L)','脑岛(R)','前扣带和旁扣带脑回(L)','前扣带和旁扣带脑回(R)','内侧和旁扣带脑回(L)','内侧和旁扣带脑回(R)','后扣带回(L)','后扣带回(R)','海马(L)','海马(R)','海马旁回(L)','海马旁回(R)','杏仁核(L)','杏仁核(R)','距状裂周围皮层(L)','距状裂周围皮层(R)','楔叶(L)','楔叶(R)','舌回(L)','舌回(R)','枕上回(L)','枕上回(R)','枕中回(L)','枕中回(R)','枕下回(L)','枕下回(R)','梭状回(L)','梭状回(R)','中央后回(L)','中央后回(R)','顶上回(L)','顶上回(R)','顶下缘角回(L)','顶下缘角回(R)','缘上回(L)','缘上回(R)','角回(L)','角回(R)','楔前叶(L)','楔前叶(R)','中央旁小叶(L)','中央旁小叶(R)','尾状核(L)','尾状核(R)','豆状壳核(L)','豆状壳核(R)','豆状苍白球(L)','豆状苍白球(R)','丘脑(L)','丘脑(R)','颞横回(L)','颞横回(R)','颞上回(L)','颞上回(R)','颞极：颞上回(L)','颞极：颞上回(R)','颞中回(L)','颞中回(R)','颞极：颞中回(L)','颞极：颞中回(R)','颞下回(L)','颞下回(R)'};
a=[1:2:90];
region=region(a);
region_l=cell(1,45);
for i=1:45
b=region(i);
b=string(b);
c=erase(b,'.L');
region_l{i}=c;
end


X=X_data;
testlabel=Y(index);
Y(find(Y~=0))=1;
tag=0;
numF=50;
test_h0_label = tag;
X_train = double(X);
Y(index)=tag;
Y_temp= nominal(ismember(Y,0)); 
Y_train = (double(Y_temp)-1)*2-1;   
%  labels:  neg_class: -1, pos_class: +1
    %selected FC number . Important parameter!

    switch lower(selection_method)
        case 'rfe'
            [ranking,score] = SVM_RFE_score(X_train,Y_train,numF);
            
    end
gra=zeros(90,8);
gra=gra(:);
gra(ranking)=score;
aaa=reshape(gra,[90,8]);
aa=aaa';
aa(aa~=0)=1;
imagesc(aa);
% colormap(flipud(gray));
colorbar;
box off;
xticks(1:2:90);  
xticklabels(region_l);  % 设置刻度标签
% 设置坐标轴刻度线
ax = gca;
ax.XAxis.TickLength = [0 0];
ax.YAxis.TickLength = [0 0];

hold on
for i=[0:8.5]
line([0,91],[i+0.5,i+0.5],'color','k','LineWidth', 0.1);
end
for i=[0:90]
line([i+0.5,i+0.5],[0,10],'color','k','LineWidth', 0.1);
end
% for i=[1:2:90]
% line([i+0.5,i+0.5],[0,9.5],'color','k','LineWidth', 0.1);
% end

box off;

AAA=sum(aa,1);
lim_index=[1:90];
[ranked_score,index]=sort(AAA,'descend');
brain_index=lim_index(index);
raned_score=round(ranked_score,4);

function [r_out ,score]= SVM_RFE_score(data,label,numf)
   Num = size(data,2);
   s = [1:Num];

while ~isempty(s)
            X = data(:,s);

           tmp =  size(s,2)-numf;
            if tmp>0    % 继续排序           
               model = fitcsvm(X,label);   %by tyb
               w = model.SupportVectors' * (model.SupportVectorLabels.*model.Alpha);%%求权重  nSV 是各个类的支持向量数量   在sv_coef中每一行存放该支持向量与其它各类（n-1类）进行一对一训练时系数
               c = w.^2;
               c=c';

               [~, c_indx] = sort(c,'ascend');
               num_reduce = floor(max(tmp,2)/2);  % 需要减少的组数

               tmp_c = c_indx(1:num_reduce);  % 选出组数的索引,相对索引
         
               s(tmp_c(:)) = [];    % 里面是绝对索引，妙！
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
            else
               [~,index] = sort(c,'ascend');
               c(index(1))=[];
               break;  
            end
end
   score=c;
   r_out=s;
  end




 

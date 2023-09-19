listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe
load('regionlxf.mat')
% Load the data and select features for classification
load(['ADS_all_region.mat'])
X=X_data;
Y(Y==0)=[];
Y_train=Y-1; 
numF=50;
X_train = double(X);
a=[1:2:90];
region=region(a);
region_l=cell(1,45);
for i=1:45
b=region(i);
b=string(b);
c=erase(b,'.L');
region_l{i}=c;
end

   
%  labels:  neg_class: -1, pos_class: +1
    %selected FC number . Important parameter!

    switch lower(selection_method)
        case 'rfe'
            [ranking,score] = multiSVM_RFE_score(X_train,Y_train,numF);
            
    end
gra=zeros(90,8);
gra=gra(:);
gra(ranking)=score;
aaa=reshape(gra,[90,8]);
aa=aaa';
aa(aa~=0)=1;
imagesc(aa);
colorbar;
box off;
xticks(1:2:90);  
xticklabels(region_l);  % 设置刻度标签
% 设置坐标轴刻度线
ax = gca;
ax.XAxis.TickLength = [0 0];
ax.YAxis.TickLength = [0 0];

hold on
for i=[0:8]
line([0,91],[i+0.5,i+0.5],'color','k','LineWidth', 0.1);
end
for i=[0:90]
line([i+0.5,i+0.5],[0,8.5],'color','k','LineWidth', 0.1);
end
box off;


AAA=sum(aa,1);
lim_index=[1:90];
[ranked_score,index]=sort(AAA,'descend');
brain_index=lim_index(index);


function [r_out ,score]= multiSVM_RFE_score(data,label,numf)
num=size(data,2);
s=[1:num];
numclass=size(unique(label),1);
while ~isempty(s)
      temp=[];
      tmp =  size(s,2)-numf;
     if tmp>0    % 继续排序    
            for i= 0:numclass-2
               for j=i+1:numclass-1
                 X = [data(label==i,s);data(label==j,s)];
                 l1=size(find(label==i),1);
                 l2=size(find(label==j),1);
                 temp_label=[1:(l1+l2)]'; 
                 temp_label(1:l1,:)=1;
                 temp_label(l1+1:end,:)=-1 ;               
                 model = fitcsvm(X,temp_label);   %by tyb
                 w = model.SupportVectors' * (model.SupportVectorLabels.*model.Alpha);%%求权重  nSV 是各个类的支持向量数量   在sv_coef中每一行存放该支持向量与其它各类（n-1类）进行一对一训练时系数
                 c = w.^2;
                 c=c';
                 temp=[temp;c];
               end
            end
            temp=sum(temp);
            [~, c_indx] = sort(temp,'ascend');
            temp(c_indx(1))=[];
            num_reduce = floor(max(tmp,2)/2);
            tmp_c = c_indx(1:num_reduce);
            s(tmp_c(:)) = [];   
             
     else
%                [~,index] = sort(temp,'ascend');
%                temp(index(1))=[];
            break
     end
     r_out=s;
     score=temp;
end
  
end




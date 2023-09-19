listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe
index=1;
% Load the data and select features for classification
load(['ADS_limbic.mat'])
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
gra=zeros(22,8);
gra=gra(:);
gra(ranking)=score;
aaa=reshape(gra,[22,8]);
AAA=sum(aaa,2);
lim_index=[21,22,31,32,33,34,35,36,37,38,39,40,41,42,71,72,73,74,75,76,77,78];
[~,index]=sort(AAA,'descend');
brain_index=lim_index(index);

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




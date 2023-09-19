function r_out = multi_SVM(data,label,numf)%label要从0开始，numf为要选择的前多少特征数，data为样本数×特征数

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
            num_reduce = floor(max(tmp,2)/2);
            tmp_c = c_indx(1:num_reduce);
            s(tmp_c(:)) = [];   
        else
            break
     end
     r_out=s;
end
  
end
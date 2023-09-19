function r_out = SVM_RFE_tyb_fast(data,label,numf)
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
               break;  
            end
 end

   r_out=s;
  end

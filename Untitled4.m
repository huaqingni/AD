addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% 6 fixed region
feature_fix = [10 38 42 63 68 90];    % 4 datasets excluding PK_1
% feature_fix = [ 9 42 63 68 69 75 79 90];    % 5 datasets

% data_name = 'BPeking_data';
% index = 3;

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};

% [ methodID ] = readInput( listFS );
selection_method = listFS{10}; % Selected rfe

% Load the data and select features for classification
% load fisheriris
load(['BNI_data.mat'])
% X_temp = conny;   �
X_temp = grad;                 
Y_temp = a;
X = [];

for i=1:max(size(X_temp))    
   temp = X_temp{1,i}';
   X = [X; temp];
end

% X = meas; clear meas
% Extract the Setosa class
index=1
Y = nominal(ismember(Y_temp,1))
X_train = double(X);
Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h0_label = double(Y(index));

%numF = size(X_train,2);
% numF = 110;   %tyb 2020-9-4
numF_1 = 112;   %tyb 2022-11-2   64/8 = 10������
numF = 50;     % 64���ݶȳɷ���ѡ50
num_group=14
num_in_group=8
data=X_train;
label=Y_train;
     numclass = 2;
    Num = size(data,2);
    l_fix = length(feature_fix);
    
    s = [1:Num];    % 存放每次选剩下来的特征的绝对索引！！！！
    s_tmp = s;
    r = [];
    r_tmp = r;
    iter = 1;  %  删除的特征个数

        % grouped SVM
        while ~isempty(s)
            X = data(:,s);
            [~,ia,~] = intersect(s,feature_fix);
            % model = libsvmtrain(label, X);
            tmp =  size(s,2)/num_in_group - num_group;
        
            if tmp>0    % 继续排序           
               model = fitcsvm(X,label);   %by tyb
               w = model.SupportVectors' * (model.SupportVectorLabels.*model.Alpha);%%求权重  nSV 是各个类的支持向量数量   在sv_coef中每一行存放该支持向量与其它各类（n-1类）进行一对一训练时系数
               c = w.^2;
               if num_in_group >1
                 c_r = reshape(c,length(c)/num_in_group,num_in_group);
                 c_r_1 = sum(c_r');
               else
                   c_r = c;
                   c_r_1 = c;
               end
               
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % 折半去除
               d_R = [c_r_1; s(1:length(c_r_1))];   % 构造数据结构
               d_R(:,ia)=[];                        % 去除固定脑区
               
%                [~, c_indx] = sort(c_r_1,'ascend');
               [~, c_indx] = sort(d_R(1,:),'ascend');
               
               
               num_reduce = floor(max(tmp,2)/2);  % 需要减少的组数
               

               
               tmp_c = c_indx(1:num_reduce);  % 选出组数的索引
               aa=s(1:length(c_r_1))
               bb=d_R(2,tmp_c)
               [~,~,tmp_c] = intersect(d_R(2,tmp_c),s(1:length(c_r_1)));  % 对应在s上是第几个
               tmp_c = tmp_c';
               
               if num_in_group==1
%                    tmp_c = tmp_c';
               else
                   tmp_c = tmp_c'*ones(1,num_in_group) + ones(length(tmp_c),1)*[0:num_in_group-1]*size(c_r,1);   % 相对索引
               end
               tmp_c = tmp_c';
               r = [s(tmp_c(:)),r];
               s(tmp_c(:)) = [];    % 里面是绝对索引，妙！
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
            else
               % 满足要求后，直接填充r 
               s_r = reshape(s,length(s)/num_in_group,num_in_group);
               s_1 = s_r';          % 按每隔8个为一组
               r = [s_1(:)' r];      
               break;  
            end
        end

         r_out = r(1:num_in_group:end); 
          isB = ismember(r_out(1:numF_1/8), feature_fix);
         spe = r_out(~isB);%提取出二分时的特定脑区
         ranking = reshape([1:720],90,8);
            ranking = ranking(spe,:);
  

            ranking = ranking(:);
            sub_r_out = SVM_RFE_tyb_fast(Y_train, X_train(:,ranking), 4,8);
            com=[feature_fix,spe(sub_r_out(1:4))]
            ranking = reshape([1:720],90,8);
              ranking = ranking(com,:);
              ranking = ranking(:);
              r_out = SVM_RFE_tyb_fast(Y_train, X_train(:,ranking), numF,1);
              ranking= ranking(r_out);

            
            
            
            
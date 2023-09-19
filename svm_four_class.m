function [train_h_data, train_h_label ,tag_num,testlabel] = svm_four_class(index, tag,numF) 
% tag表示假设的标签
% Include dependencies
addpath('./lib');                      % dependencies
addpath('./methods');            % FS methods
addpath(genpath('./lib/drtoolbox'));

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe

% Load the data and select features for classification
load(['ADS_all_region.mat'])
X=X_data;
testlabel=Y(index);
Y(Y==0)=[];
if index>108
    X=X(109:end,:);
    Y=Y;
    index=index-108;
  [train_h_data, train_h_label] = train_h(index, X, Y, selection_method,tag,numF);
  [train_h_data, train_h_label] = energy_normalization(train_h_data, train_h_label);
else
  Y(end)=[];
  X(end,:)=[];
  X=X([index,109:end],:);
  Y=[1;Y];
  index=1;
  [train_h_data, train_h_label] = train_h(index, X, Y, selection_method,tag,numF);
  [train_h_data, train_h_label] = energy_normalization(train_h_data, train_h_label);
end
   [~,nn_1]=find(train_h_label==1);
   [~,nn_2]=find(train_h_label==2);
   [~,nn_3]=find(train_h_label==3);
   [~,nn_4]=find(train_h_label==4);
   tag_num = [ sum(nn_4); sum(nn_3); sum(nn_2);sum(nn_1)];

end




function [train_data_out, train_label_out] = energy_normalization(train_data, train_label)

    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label==1));
    avg_energy_2 = mean(sample_energy_tmp(train_label==2));
    agv_energy_3 = mean(sample_energy_tmp(train_label==3));
    avg_energy_4 = mean(sample_energy_tmp(train_label==4));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label==1) = agv_energy_1;
    sample_energy_map(train_label==2) = avg_energy_2;
    sample_energy_map(train_label==3) = agv_energy_3;
    sample_energy_map(train_label==4) = avg_energy_4;
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
   
    train_data_out = (tmp ./ energy_map)'; 
    train_label_out = train_label;
    
    return
end

function [train_h_data, train_h_label] = train_h(index, X, Y, selection_method,tag,numF)
test_h_label = tag;
X_train = double(X);
Y(index)=tag;
Y_train=Y-1;  %多分SVM标签从0开始


    switch lower(selection_method)
        case 'rfe'
            ranking = multi_SVM(X_train,Y_train,numF);
    end


    k = numF;     
    train_h_data = X_train(:,ranking(1:k));
    train_h_label = Y;
    train_h_data(index,:)=[];
    train_h_label(index)=[];

    [~,indx_1] = sort(train_h_label,'descend');
    train_h_label= train_h_label(indx_1);
    train_h_data = train_h_data(indx_1,:);
    return
end

function [ranking] = train_h0(index, X, Y, selection_method,tag)
Y(find(Y~=0))=1;
test_h0_label = tag;
X_train = double(X);
Y(index)=tag;
Y_temp= nominal(ismember(Y,0)); 
Y_train = (double(Y_temp)-1)*2-1;   
%  labels:  neg_class: -1, pos_class: +1
numF = 60;      %selected FC number . Important parameter!

    switch lower(selection_method)
        case 'rfe'
            ranking = SVM_RFE_tyb_fast(X_train,Y_train,numF);
    end


    k = numF;     
    train_h0_data = X_train(:,ranking(1:k));
    train_h0_label = Y;

    train_h0_label= logical(train_h0_label);
    train_h0_data = train_h0_data;

    return
end



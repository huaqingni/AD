index=287;
% Include dependencies
addpath('./lib');                      % dependencies
addpath('./methods');            % FS methods
addpath(genpath('./lib/drtoolbox'));

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe

% Load the data and select features for classification
load(['ASD.mat'])
X=X;
testlabel=Y(index);
[ranking] = train_h0(index, X, Y, selection_method,1);
Y(Y==0)=[];
X=X(109:end,ranking);
Y=Y;
index=index-108;
[train_h_data, train_h_label,ranking1] = train_h(index, X, Y, selection_method,1);
[train_h_data, train_h_label,ranking2] = train_h(index, X, Y, selection_method,2);
[train_h_data, train_h_label,ranking3] = train_h(index, X, Y, selection_method,3);
[train_h_data, train_h_label,ranking4] = train_h(index, X, Y, selection_method,4);
rank=[ranking1;ranking2;ranking3;ranking4;]







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

function [train_h_data, train_h_label,ranking] = train_h(index, X, Y, selection_method,tag)
test_h_label = tag;
X_train = double(X);
Y(index)=tag;
Y_train=Y-1;  %多分SVM标签从0开始
%  labels:  neg_class: -1, pos_class: +1
numF = 20;      %selected FC number . Important parameter!

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



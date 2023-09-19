
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

data_name = 'BPeking_data';
index = 3;

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
listSets = {'BNYU_data','BPeking_data','BPeking_1_data','BKKI_data','BNI_data'};

selection_method = listFS{10}; % Selected rfe
feature_fix = [10 38 42 63 68 90]; 
rank=reshape([1:720],90,8);
ranking=(rank(feature_fix,:))
ranking=ranking(:)'

X =[];
Y= [];
Y_sub=[]
% load all data
for ii = [1 2  4 ]
    load([listSets{ii} '.mat']);
    X_temp = grad;               
    Y_temp = a;
    Ytemp=tag;
    
   for i = 1:max(size(X_temp))    
       temp = X_temp{1,i}';
       X = [X; temp];   
   end
   
   
   Y_sub=[Y_sub;Ytemp]
end
Y_s=Y_sub(Y_sub~=0);
X=X(Y_sub~=0,ranking)


Y_AD=[]
for i=1:max(size(Y_s))
    if Y_s(i)==1
        Y_AD=[ Y_AD;1];
    else
        Y_AD=[ Y_AD;0];
    end
end

Y=logical(Y_AD);



Y = nominal(ismember(Y,1));


X_train = double(X);
Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h0_label = double(Y(index));

%numF = size(X_train,2);
% numF = 110;   %tyb 2020-9-4
numF_1 = 24   %tyb 2022-11-2   64/8 = 10������
numF = 50;     % 64���ݶȳɷ���ѡ50���ɷ�

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; %For unsupervised feature selection, you should tune
            %this parameter k, the default k is 5.
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
%             ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
%             r_out = SVM_RFE_tyb(Y_train, X_train);
             
             D_record = [];
             % 检测稳定的显著脑区！！
             for i = 1:length(Y_train)
              Y_train_1 = Y_train;
              Y_train_1(i) = [];
              X_train_1 = X_train;
              X_train_1(i,:) = [];
              r_out = SVM_RFE_tyb_fast(Y_train_1, X_train_1, numF_1/8,8);
              ranking = reshape([1:720],90,8);
              ranking = ranking(r_out(1:numF_1/8),:);
%               ranking = ranking(:);
              D_record = [D_record ranking(:,1)];
             end 
              
             save('tmp_sub.mat','D_record')
              
              r_out = SVM_RFE_tyb_fast(Y_train, X_train(:,ranking), numF,1);
              ranking = ranking(r_out);


%             load('r_out_80.mat');
%             ranking_indx = spider_wrapper(X_train(:,ranking_tmp),Y_train,numF,lower(selection_method));
%             ranking_tmp_1 = ranking_tmp(ranking_indx);

        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;    % default, it should be cross-validated.
            sup = 1;        % Supervised or Not
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; % default, it should be cross-validated.
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end

   % k = 110; % select the first 110 features
    k = numF; % select the first 55 features

    %svmStruct = fitcsvm(X_train(:,ranking<=k),Y_train,'Standardize',true,'KernelFunction','RBF',...
    %'KernelScale','auto','OutlierFraction',0.0);

    %C = predict(svmStruct,X_train(:,ranking<=k));
    %err_rate = sum(Y_train~= C)/max(size(Y_train)); % mis-classification rate
    %% conMat = confusionmat(Y_test,C); % the confusion matrix
    %fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',...
    %    selection_method,100*(1-err_rate),err_rate);
   
 
    train_h0_data = X_train(:,ranking(1:k));
    train_h0_label = Y_temp;

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];

    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= train_h0_label(indx_1);
    train_h0_data = train_h0_data(indx_1,:);




load('ADS_all_region')
Y(Y~=0)=-1;
Y(Y==0)=1;
r=SVM_RFE_tyb_fast(Y,X_data,50,8);


function r_out = SVM_RFE_tyb_fast(label, data, num_group, num_in_group)

% ��dataΪ��ά����ʱ
% num_in_group: �����м������� size��data��2��
% num_group������  size��data��1��
% r_out����������Ҫ�̶����

    % SVM-RFE
    % SVM Recursive Feature Elimination (SVM RFE)
    % by liyang @BNU Math
    % Email:patrick.lee@foxmail.com
    % last modified 2010.09.18
    
%     data = randn(216,720);
%     label = randn(216,1)>0;
%     numclass=size(unique(label),1);
    numclass = 2;
    Num = size(data,2);
    
    
    s = [1:Num];
    s_tmp = s;
    r = [];
    r_tmp = r;
    iter = 1;  %  ɾ������������

    if numclass==2

        
        % grouped SVM
        while ~isempty(s)
            X = data(:,s);
            % model = libsvmtrain(label, X);
            tmp =  size(s,2)/num_in_group - num_group;
        
            if tmp>0    % ��������           
               model = fitcsvm(X,label);   %by tyb
               w = model.SupportVectors' * (model.SupportVectorLabels.*model.Alpha);%%��Ȩ��  nSV �Ǹ������֧����������   ��sv_coef��ÿһ�д�Ÿ�֧���������������ࣨn-1�ࣩ����һ��һѵ��ʱϵ��
               c = w.^2;
               if num_in_group >1
                 c_r = reshape(c,length(c)/num_in_group,num_in_group);
                 c_r_1 = sum(c_r');
               else
                   c_r = c;
                   c_r_1 = c;
               end
%                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                [~ , f_1] = min(c_r_1);
%                f = f_1:size(c_r,1):length(c);
%                r = [s(f),r];
%                s(f) = [];    % �����Ǿ����������
%                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % �۰�ȥ��
               [~, c_indx] = sort(c_r_1,'ascend');
               num_reduce = floor(max(tmp,2)/2);  % ��Ҫ���ٵ�����
%                num_reduce = 1;
               tmp_c = c_indx(1:num_reduce);  % ѡ������������,�������
               if num_in_group==1
%                    tmp_c = tmp_c';
               else
                   tmp_c = tmp_c'*ones(1,num_in_group) + ones(length(tmp_c),1)*[0:num_in_group-1]*size(c_r,1);
               end
               tmp_c = tmp_c';
               r = [s(tmp_c(:)),r];
               s(tmp_c(:)) = [];    % �����Ǿ����������
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
            else
               % ����Ҫ���ֱ�����r 
               s_r = reshape(s,length(s)/num_in_group,num_in_group);
               s_1 = s_r';          % ��ÿ��8��Ϊһ��
               r = [s_1(:)' r];      
               break;  
            end
        end

         r_out = r(1:num_in_group:end);   % r��8��һ�飡
                

        end
%         display(iter);
    end

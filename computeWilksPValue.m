function [p,lambda]= computeWilksPValue(data1, data2)
    % data1: n1 x 3 matrix, 第一组数据，每行表示一个样本，三个维度
    % data2: n2 x 3 matrix, 第二组数据，每行表示一个样本，三个维度
%     data1=rand(170,3);
%     data2=ones(25,3);
    % 合并数据
    data = [data1; data2];
    n1=size(data1,1);
    n2=size(data2,1);

    group = [ones(n1, 1); 2*ones(n2, 1)];

% 执行多元方差分析（MANOVA）
[a, p, stats] = manova1(data, group);

% 提取Wilks' Lambda的值
lambda = stats.lambda;

% 提取自由度
df1 = stats.dfB;
df2 = stats.dfT;

% 计算Wilks' Lambda的P值
p_value = calculateWilksLambdaPValue(lambda, df1, df2);  
end
function p_value = calculateWilksLambdaPValue(lambda, df1, df2)
    % lambda: Wilks' Lambda值
    % df1: 分子自由度
    % df2: 分母自由度

    % 计算F统计量
    f_stat = (1 - lambda) / (lambda * (df2 - df1));

    % 计算P值
    p_value = 1 - fcdf(f_stat, df1, df2);
end

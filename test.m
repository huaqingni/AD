% 假设你的数据存储在向量 data 中
data = [0.5,0.8,1.2, 1.5, 1.6, 1.8, 2.1, 2.3, 2.4, 2.5, 2.6, 2.8];

% 使用高斯分布进行拟合
pd = fitdist(data', 'Normal');

% 获取拟合的参数
mu = pd.mu;  % 均值
sigma = pd.sigma;  % 标准差

% 绘制原始数据的直方图
histogram(data, 'Normalization', 'pdf');
hold on;

% 绘制拟合的高斯分布曲线
x = linspace(min(data), max(data), 100);
y = pdf(pd, x);
plot(x, y, 'r', 'LineWidth', 2);

% 添加图例和标签
legend('数据分布', '高斯拟合');
xlabel('数据');
ylabel('概率密度');

% 显示图形
hold off;

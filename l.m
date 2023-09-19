% x=0:1:4;
%  a=[97.96 	100.00 	95.74 	96.24 	98.08 ];
%  b=[98.47 	98.29 	98.69 	98.91 	98.59 ]; 
%  c=[98.27 	98.64 	98.13 	95.13 	96.81 ]; 
%  d=[95.53 	95.78 	95.36 	93.39 	94.53 ]; 
%  e=[97.16 	97.33 	97.10 	93.01 	95.07 ]; 
%  plot(x,a,'.-b');
%  hold on;
%  plot(x,b,'.-r');
%  hold on;
%  plot(x,c,'.-y');
% hold on;
% plot(x,d,'.-g');
%  hold on;
%  plot(x,e,'.-black');
%  hold on;
% axis([0,4,90,100])  %确定x轴与y轴框图大小
% set(gca,'XTick',[0:1:4]) %x轴范围1-6，间隔1
% set(gca,'XTickLabel',{'Accuarcy', 'Sensitivity', 'Specifity', 'Precision','F1 score'})
% set(gca,'YTick',[90:1:100]) %y轴范围0-700，间隔100
% legend('NI','KKI','NYU','PU','PU_1');   %右上角标注
% xlabel('')  %x轴坐标描述
% ylabel('') %y轴坐标描述

clear
clc
Y=[97.96,100.00,95.74
    98.47,98.29,98.69
    98.27,98.64,98.13
    95.53,95.78,95.36
    97.16,97.33,97.10];
figure
set(gcf,'position',[100,100,800,500])   %调整图的位置
X=1:5;
h=bar(X,Y,1.0);       %画出两个直方图，宽度0.9，可调
set(gca,'XTickLabel',{'NI','NYU','KKI','PU','PU_1'},'FontSize',12,'FontName','宋体'); %修改横坐标名称、字体
set(h(1),'FaceColor',[0 0.7 1])     % 设置条形图颜色，图1
set(h(2),'FaceColor',[1 0.8 0])     % 设置条形图颜色，图2
set(h(3),'FaceColor',[1 0.0 0])
ylim([95,100]);      %y轴刻度
%修改x,y轴标签，中英文字体分开
ylabel('\fontname{宋体}\fontsize{12}百分比');
xlabel('\fontname{宋体}\fontsize{12}数据库'); 
%修改图例，中英文字体分开
legend({'Accuarcy','Sensitivity','Specifity'}, 'FontSize',12);
set(gca,'xtick',1:5);   %x轴刻度
Y_1=roundn(Y,-4);   %调整y轴数字的精度，保留小数点后几位
%在柱状图上标数字
for i = 1:length(X)
    text(X(i)-0.2,Y_1(i,1),num2str(Y_1(i,1)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10,'FontName','Times New Roman');
    text(X(i),Y_1(i,2),num2str(Y_1(i,2)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10,'FontName','Times New Roman');
    text(X(i)+0.2,Y_1(i,3),num2str(Y_1(i,3)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10,'FontName','Times New Roman');
end

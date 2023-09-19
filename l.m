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
% axis([0,4,90,100])  %ȷ��x����y���ͼ��С
% set(gca,'XTick',[0:1:4]) %x�᷶Χ1-6�����1
% set(gca,'XTickLabel',{'Accuarcy', 'Sensitivity', 'Specifity', 'Precision','F1 score'})
% set(gca,'YTick',[90:1:100]) %y�᷶Χ0-700�����100
% legend('NI','KKI','NYU','PU','PU_1');   %���ϽǱ�ע
% xlabel('')  %x����������
% ylabel('') %y����������

clear
clc
Y=[97.96,100.00,95.74
    98.47,98.29,98.69
    98.27,98.64,98.13
    95.53,95.78,95.36
    97.16,97.33,97.10];
figure
set(gcf,'position',[100,100,800,500])   %����ͼ��λ��
X=1:5;
h=bar(X,Y,1.0);       %��������ֱ��ͼ�����0.9���ɵ�
set(gca,'XTickLabel',{'NI','NYU','KKI','PU','PU_1'},'FontSize',12,'FontName','����'); %�޸ĺ��������ơ�����
set(h(1),'FaceColor',[0 0.7 1])     % ��������ͼ��ɫ��ͼ1
set(h(2),'FaceColor',[1 0.8 0])     % ��������ͼ��ɫ��ͼ2
set(h(3),'FaceColor',[1 0.0 0])
ylim([95,100]);      %y��̶�
%�޸�x,y���ǩ����Ӣ������ֿ�
ylabel('\fontname{����}\fontsize{12}�ٷֱ�');
xlabel('\fontname{����}\fontsize{12}���ݿ�'); 
%�޸�ͼ������Ӣ������ֿ�
legend({'Accuarcy','Sensitivity','Specifity'}, 'FontSize',12);
set(gca,'xtick',1:5);   %x��̶�
Y_1=roundn(Y,-4);   %����y�����ֵľ��ȣ�����С�����λ
%����״ͼ�ϱ�����
for i = 1:length(X)
    text(X(i)-0.2,Y_1(i,1),num2str(Y_1(i,1)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10,'FontName','Times New Roman');
    text(X(i),Y_1(i,2),num2str(Y_1(i,2)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10,'FontName','Times New Roman');
    text(X(i)+0.2,Y_1(i,3),num2str(Y_1(i,3)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',10,'FontName','Times New Roman');
end

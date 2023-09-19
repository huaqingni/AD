
load('all_region_roc.mat')
roc0=table2array(roc0);
roc1=table2array(roc1);
roc2=table2array(roc2);
roc3=table2array(roc3);
roc4=table2array(roc4);

[hctpr,hcauc]=tprcal(roc0);
[GADtpr,GADauc]=tprcal(roc1);
[KGtpr,KGauc]=tprcal(roc2);
[PDtpr,PDauc]=tprcal(roc3);
[SADtpr,SADauc]=tprcal(roc4);
h1 = plot([[0],0:0.0001:1], [0;hctpr],'-r', 'linewidth',2);
hold on
h1 = plot([[0],0:0.0001:1], [0;KGtpr],':m', 'linewidth',2);
h1 = plot([[0],0:0.0001:1], [0;PDtpr],':g', 'linewidth',2);
h1 = plot([[0],0:0.0001:1], [0;GADtpr],'-.k', 'linewidth',2);
h1 = plot([[0],0:0.0001:1], [0;SADtpr],':b', 'linewidth',2);

ylabel('True positive rate')
xlabel('False positive rate')
%legend('ROC curve of HC  (AUC =0.9981 )','ROC curve of ADHD-I  (AUC=0.9528)','ROC curve of ADHD-C  (AUC =0.9619 )')      % NYU
 legend('HC,AUC =0.9669','SP,AUC =0.9737','PD,AUC =0.9846','GAD,AUC =0.9627','SAD,AUC =0.9774')  %Peking


axes('position',[0.32,0.4,0.4,0.4]);     %关键在这句！所画的小图
h1 = plot([[0],0:0.0001:1], [0;hctpr],'-r', 'linewidth',2);
hold on
h1 = plot([[0],0:0.0001:1], [0;GADtpr],'-.k', 'linewidth',2);

h1 = plot([[0],0:0.0001:1], [0;KGtpr],':m', 'linewidth',2);
h1 = plot([[0],0:0.0001:1], [0;PDtpr],':g', 'linewidth',2);
h1 = plot([[0],0:0.0001:1], [0;SADtpr],':b', 'linewidth',2);

axis([0 0.10 0.8 1])
set(gca,'XTick',0:0.02:0.10);
set(gca,'YTick',0.8:0.05:1);
hold on

function [tpr_agv,auc]=tprcal(roc0)
tpr_agv = zeros(10001,1);
tpr = zeros(10001,1);
tpr(10001)=1;
for i = 1:30
a=round(roc0(i,2)*10000);
tpr(1:a)=0;
tpr(a+1:10000)=roc0(i,1);
tpr_agv=tpr_agv+tpr;
end
tpr_agv = tpr_agv/30;
auc=sum(tpr_agv)/10000
return
end

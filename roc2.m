
load('all_region_roc.mat')
roc0=table2array(roc0);

[hctpr,hcauc]=tprcal(roc0);

h1 = plot([[0],0:0.0001:1], [0;hctpr],'-r', 'linewidth',2);
hold on


ylabel('True positive rate')
xlabel('False positive rate')
%legend('ROC curve of HC  (AUC =0.9981 )','ROC curve of ADHD-I  (AUC=0.9528)','ROC curve of ADHD-C  (AUC =0.9619 )')      % NYU

axis([0 1 0 1])

axes('position',[0.25,0.4,0.3,0.3]);     %关键在这句！所画的小图
h1 = plot([[0],0:0.0001:1], [0;hctpr],'-r', 'linewidth',2);
hold on
axis([0 0.10 0.7 1])
set(gca,'XTick',0:0.03:0.15);
set(gca,'YTick',0.7:0.05:1);
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

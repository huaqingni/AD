load('regional_ALFFMap_HC_1.mat')
HC=ALFF;
l1=zeros(108,1);
load('regional_ALFFMap_GAD_1.mat')
GAD=ALFF;
l2=ones(48,1);
load('regional_ALFFMap_Konggao_1.mat')
KG=ALFF;
l3=2*ones(55,1);
load('regional_ALFFMap_PD_1.mat')
PD=ALFF;
l4=3*ones(51,1);
load('regional_ALFFMap_SAD_1.mat')
SAD=ALFF;
l5=4*ones(25,1);
X=[HC;GAD;KG;PD;SAD];
Y=[l1;l2;l3;l4;l5];

save('ASD','X','Y');


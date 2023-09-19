
load('regional_RehoMap_HC_1.mat')
HC=Reho;
load('regional_RehoMap_GAD_1.mat')
GAD=Reho;
load('regional_RehoMap_Konggao_1.mat')
KG=Reho;
load('regional_RehoMap_PD_1.mat')
PD=Reho;
load('regional_RehoMap_SAD_1.mat')
SAD=Reho;
X1=[HC;GAD;KG;PD;SAD]

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
X2=[HC;GAD;KG;PD;SAD];
Y=[l1;l2;l3;l4;l5];

load('regional_VBMMap_HC_1.mat')
HC=VBM;
load('regional_VBMMap_GAD_1.mat')
GAD=VBM;
load('regional_VBMMap_Konggao_1.mat')
KG=VBM;
load('regional_VBMMap_PD_1.mat')
PD=VBM;
load('regional_VBMMap_SAD_1.mat')
SAD=VBM;
X3=[HC;GAD;KG;PD;SAD] ;
X=cell(1,287);
for i=1:287
    a=[X1(i,1:90);X2(i,1:90);X3(i,1:90)];
    X{i}=a;
end
save('ASD','X','Y');













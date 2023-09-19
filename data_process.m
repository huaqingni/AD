load('ASD.mat')
X_gra=X;
for i=1:max(size(X))
    X{i}= normalize(X{i},2)
    X{i}= corr(X{i});  
    gm_m = GradientMaps();
%     gm_m = GradientMaps('kernel','g','approach','pca','alignment','');

    gm= gm_m.fit(X{i});
    X{i}=gm.gradients{1};

end


save('ADS_gra.mat','X','Y');

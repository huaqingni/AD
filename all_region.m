load('ADS_gra.mat');
for i=1:max(size(X))
   X{i}=X{i}(:);
end
 X_data=[];
 for i=1:max(size(X))
     a=[X{i}(:)]';
     X_data=[X_data;a];
 end
save('ADS_all_region.mat','X_data','Y');
load('ADS_gra.mat');
index=[21,22,31,32,33,34,35,36,37,38,39,40,41,42,71,72,73,74,75,76,77,78];

for i=1:max(size(X))
   X{i}=X{i}(index,:);
end
 X_data=[];
 for i=1:max(size(X))
     a=[X{i}(:)]';
     X_data=[X_data;a];
 end
save('ADS_limbic.mat','X_data','Y');
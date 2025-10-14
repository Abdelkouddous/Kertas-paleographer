%chaincode local_FE + labeling
clc; 
clear;
tic;

repsource = '/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/training';
xt = '*.jpg'; % sinon jpg
chemin = fullfile(repsource, xt);
list =  dir(chemin);
nfile = length(list);
label_training = zeros(42,1);
features_training = zeros(240);
 disp('___processing ___'); 

    for i=1:nfile
        inf = list(i).name; 
       
      %{
      %section 1 labeling
      label_training(i,:)=labeling(inf);
      csvwrite('label_training_ChainCodeLocalFE.csv', label_training)
      histogram(label_training);
      %section 2 feature extracting
      %}
      features_training (i,:) = ChainCode_LocalFeatures_Extraction_PRL(inf);
     % if isnan(features_training)
       %   features_training(i,:)=abs(rand(0.01,0.3)*cos(i));
      %end
      %save(['Features.mat'], features_training);
      csvwrite('features_training_ChainCodeLocalFE.csv', features_training);      
     end    
toc;
%chaincode local FE training
clc; 
clear all;
tic;

repsource = '/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/training';
xt = '*.jpg'; % sinon jpg
chemin = fullfile(repsource, xt);
list =  dir(chemin);
nfile = length(list);
label_training = zeros(42,1);
features_training = zeros(624);

    for i=1:nfile
        inf = list(i).name; 
        disp('___processed');
      %section 1 labeling
      label_training(i,:)=labeling(inf);
      csvwrite('label_training_ChainCodeLocalFE.csv', label_training)
      %histogram(label_training);
      %section 2 feature extracting
      features_training (i,:) = ChainCode_GlobalFeatures_Extraction_PRL(inf);
      csvwrite('features_training_ChainCodeLocalFE.csv', features_training)      
     end    
toc;
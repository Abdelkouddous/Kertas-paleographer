%Polygon_FE_training
clc; 
clear;
tic;

repsource = '/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/training';
xt = '*.jpg'; % sinon jpg
chemin = fullfile(repsource, xt);
list =  dir(chemin);
nfile = length(list);
%label_testing = zeros(42,1);
features_training = zeros(42);
fprintf('Extracting features for Polygon training dataset ...')
    for i=1:nfile
        inf = list(i).name; 
        disp(' ___processed');
      %{
      %section 1 labeling
      label_testing(i,:)=labeling(inf);
      csvwrite('label_testing.csv', label_testing)
      histogram(label_testing);
      %section 2 feature extracting
        %}
      features_training (i,:) = Polygon_Features_Extraction_PRL(inf);
      if isnan(features_training)
         features_training(i,:)=abs(rand(0.05,0.7)*cos(i));
      end
      csvwrite('features_training_PolygonFE.csv', features_training);      
     end    
toc;
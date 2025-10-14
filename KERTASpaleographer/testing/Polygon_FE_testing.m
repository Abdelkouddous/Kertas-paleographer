%Polygon_FE_testing
%chaincode GLOBAL FE testing
clc; 
clear;
tic;

repsource = '/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/testing';
xt = '*.jpg'; % sinon jpg
chemin = fullfile(repsource, xt);
list =  dir(chemin);
nfile = length(list);
%label_testing = zeros(42,1);
features_testing = zeros(42);

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
      features_testing (i,:) = Polygon_Features_Extraction_PRL(inf);
      if isnan(features_testing)
         features_testing(i,:)=abs(rand(0.05,0.7)*cos(i));
      end
      csvwrite('features_testing_PolygonFE.csv', features_testing);      
     end    
toc;
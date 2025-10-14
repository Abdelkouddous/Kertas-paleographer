function [] = ChainCode_LocalFeatures_Extraction_PRL
clc
clear
WINDOW_SIZE=13;
CURVATURE_BINS=11;
BINS=10;

SIG=0;


K = 0;
for i1=300:-1:1
    for j1=3:3
    X = [];

%    Fichier=['C:\Handwritings_Databases\IAM\Binarized_IAM\',int2str(i1),'_', int2str(j1),'.png']
%    Fichier1=['C:\Handwritings_Databases\IAM\Binarized_IAM\ChainLocal\',int2str(i1),'_', int2str(j1),'.mat']
    
    
    Fichier=['C:\Handwritings_Databases\CVL-Database\CVL\Writer_',int2str(i1),'_',int2str(j1),'.tif']
    Fichier1=['C:\Handwritings_Databases\CVL-Database\CVL\ChainLocal\Writer_',int2str(i1),'_',int2str(j1),'.mat']
    
    
    
%    if i1<10
%        Fichier=['C:\Handwritings_Databases\BFL_Database\CF0000',int2str(i1),'_0',int2str(j1),'.bmp']
%        Fichier1=['C:\Handwritings_Databases\BFL_Database\ChainLocal\CF0000',int2str(i1),'_0',int2str(j1),'.mat']

%    elseif (i1>=10) & (i1<100)
%        Fichier=['C:\Handwritings_Databases\BFL_Database\CF000',int2str(i1),'_0',int2str(j1),'.bmp']
%        Fichier1=['C:\Handwritings_Databases\BFL_Database\ChainLocal\CF000',int2str(i1),'_0',int2str(j1),'.mat']
%    else
%        Fichier=['C:\Handwritings_Databases\BFL_Database\CF00',int2str(i1),'_0',int2str(j1),'.bmp']
%        Fichier1=['C:\Handwritings_Databases\BFL_Database\ChainLocal\CF00',int2str(i1),'_0',int2str(j1),'.mat']
%    end

    I = imread(Fichier);  
    I = im2uint8(I);    
    I  = padarray(I,[WINDOW_SIZE WINDOW_SIZE],255);
if(SIG)
    BW=I;
else
    level = graythresh(I);
    BW = im2bw(I,level);  
end

    [labeled,numOfComponents] = bwlabel(~BW,8);
    components = regionprops(labeled,'basic');
    filteredComponents = filterComponents(BW,components,WINDOW_SIZE);
    windows = divideWritingNaturalComponent(BW,filteredComponents,WINDOW_SIZE);
    tic
    [distribution distribution2 distribution3 ]=getDirFragHistogram(BW,windows,WINDOW_SIZE,BINS);


    
    distribution = reshape(distribution,1,[]);
    distribution2 = reshape(distribution2,1,[]);
    distribution3 = reshape(distribution3,1,[]);


    X = [distribution distribution2 distribution3 ];
    save(Fichier1, 'X');  % Chaincode Based local Features.

    
end %end for ii
end % end for jj    
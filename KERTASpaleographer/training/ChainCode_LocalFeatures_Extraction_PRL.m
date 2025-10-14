function [X] = ChainCode_LocalFeatures_Extraction_PRL(Fichier)
clc
%clear
WINDOW_SIZE=9; %default 13
%CURVATURE_BINS=11;
BINS=9; %default 10

SIG=0;



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
    tic;
    [distribution,distribution2, distribution3]=getDirFragHistogram(BW,windows,WINDOW_SIZE,BINS);


    
   distribution = reshape(distribution,1,[]); %distribution = reshape(distribution,1,[]);
    distribution2 = reshape(distribution2,1,[]);
    distribution3 = reshape(distribution3,1,[]);


    X = [distribution distribution2 distribution3 ];
    
    %save('X_output', 'X');  % Chaincode Based local Features.

    
end %end for ii
 % end for jj    
 
 %filter components
 %divideWritingNaturalComposition isNoise getDirFragHistogram
 %'getChainCodedImageWithInteriorContours'
 
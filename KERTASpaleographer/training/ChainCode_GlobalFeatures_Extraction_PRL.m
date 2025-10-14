function [X] = ChainCode_GlobalFeatures_Extraction_PRL(Fichier)


clc
USE_EXTERIOR_CONTOURS_ONLY=1;
CURVATURE_BINS=2; %default 11
%WINDOW_SIZE=2; %default 13
SIG=0;

    
    I=imread(Fichier);
    I = im2bw(I, 0.95);

   I = im2uint8(I);

   
    
    if(SIG)
        bw=I;
    else
        level = graythresh(I);
        bw = imbinarize(I,level);  %bw = im2bw(I,level);
    end
        histogram = zeros(1,8)%[0 0 0 0 0 0 0 0 ];
        histogramQ = zeros(1,4)%[0 0 0 0];
        histogramOfAngles = zeros(1,8);%[0 0 0 0 0 0 0 0 ];
        histogramOfSecondGrad = zeros(1,8);%[0 0 0 0 0 0 0 0 ];
        histogramOfCurvIndex = zeros(1,CURVATURE_BINS);
        histogramOfChainCodePairs = zeros(1,64);%repmat(0,1,64);
        histogramOfChainCodeTripples = zeros(1,512);%repmat(0,1,512);    


        totalHistogramsComputed=0;%For normalization keep track of the number of hists
        [labels n] = bwlabel(~bw);%[labels n]
        components = regionprops(labels);

        components = filterComponents(components,15);
        n=length(components);
        %For each of the components
        histTime=0;
        for i=1:n
        %% Find component bounding box
             startX = components(i).BoundingBox(1)+0.5;
             startY = components(i).BoundingBox(2)+0.5;        
             w = components(i).BoundingBox(3);
             h = components(i).BoundingBox(4);
        %% Extract the component
        %Extract the component in graylevels
             cc = bw(startY:startY+h-1,startX:startX+w-1);

        %% Find contours by tracing the boundries
            cc=~cc;
            cc=cleanImage(cc);
            if(USE_EXTERIOR_CONTOURS_ONLY)
                [tempValues tempValuesQ tempAngles tempSecondGrad tempPInv fCode] = getHistogramOfChainCode(~cc,CURVATURE_BINS);
                [tempHistogramOfChainCodePairs tempHistogramOfChainCodeTripples]=getChainCodePairsHistorgram(fCode);
                histogram = histogram + tempValues;
                histogramQ = histogramQ + tempValuesQ;
                histogramOfAngles = histogramOfAngles + tempAngles;
                histogramOfSecondGrad = histogramOfSecondGrad + tempSecondGrad;
                histogramOfCurvIndex=histogramOfCurvIndex+tempPInv;
                histogramOfChainCodePairs=histogramOfChainCodePairs+tempHistogramOfChainCodePairs;
                histogramOfChainCodeTripples=histogramOfChainCodeTripples+tempHistogramOfChainCodeTripples;
            else
                [B L]=bwboundaries(cc,8);%Find boundaries of all the objects in this component
                numOfObjects = length(unique(L));%Find how many objects are there, ZEROs represent background
                objectProps=regionprops(L);%Find the properties of these objects
                    for j=1:numOfObjects-1
                        if(objectProps(j).Area>15)
                            indices= find(L==j);
                            curObject = zeros(size(cc));
                            curObject(indices) = 1;
                            %imagesc(curObject);
                            tic
                            [tempValues tempValuesQ tempAngles tempSecondGrad tempPInv fCode] = getHistogramOfChainCode(~curObject,CURVATURE_BINS);
                            [tempHistogramOfChainCodePairs tempHistogramOfChainCodeTripples]=getChainCodePairsHistorgram(fCode);
                
                            histogram = histogram + tempValues;
                            histogramQ = histogramQ + tempValuesQ;
                            histogramOfAngles = histogramOfAngles + tempAngles;
                            histogramOfSecondGrad = histogramOfSecondGrad + tempSecondGrad;
                            histogramOfCurvIndex=histogramOfCurvIndex+tempPInv;
                            histogramOfChainCodePairs=histogramOfChainCodePairs+tempHistogramOfChainCodePairs;
                            histogramOfChainCodeTripples=histogramOfChainCodeTripples+tempHistogramOfChainCodeTripples;
                            totalHistogramsComputed=totalHistogramsComputed+1;
                
                            histTime=histTime+toc;
                        end%End if
                    end%End for num of objects
            end%End if use exterior contours only
        
        end%End components loop
      
%% Normalize each histogram
     histogram = histogram./sum(histogram);
     histogramQ = histogramQ./sum(histogramQ);
     histogramOfAngles = histogramOfAngles./sum(histogramOfAngles);
     histogramOfSecondGrad = histogramOfSecondGrad./sum(histogramOfSecondGrad);
     histogramOfCurvIndex=histogramOfCurvIndex./sum(histogramOfCurvIndex);
     histogramOfChainCodePairs=histogramOfChainCodePairs./sum(histogramOfChainCodePairs);
     histogramOfChainCodeTripples=histogramOfChainCodeTripples./sum(histogramOfChainCodeTripples);

     
     X = [histogram histogramQ histogramOfAngles histogramOfSecondGrad histogramOfCurvIndex histogramOfChainCodePairs histogramOfChainCodeTripples];
     X=reshape(X,[605,1]); 
     %save('file', 'X');  % Chaincode Based Global Features.
%% 
end
%End files
%end


%%
function filteredComponents = filterComponents(components,minArea)

    %Remove the components which are too small
    numbOfComponents = length(components);
    j=1;
    while j <= numbOfComponents
     w = components(j).BoundingBox(3);
      h = components(j).BoundingBox(4);
%      if (components(j).Area < minArea)
      if (w < minArea || h<minArea)
          components(j) = [];
          numbOfComponents = length(components);         
      else
                 j=j+1;
      end

    end
    filteredComponents = components;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% FUNCTION %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%

function ICleaned = cleanImage(I)

[bw n] = bwlabel(I,8);
if n==1
    ICleaned = I; %If theres only one component, return
else
    areas = regionprops(bw,'area'); %Find the area of each component
    bigComp = find([areas.Area]==max([areas.Area])); %Find the (label of) biggest component
    idx=find(bw~=bigComp(1)); %Find the indices of all the labels other than that of bigComp
    I(idx)=0; %Remove all the other components
    ICleaned=I; 
end
%end%End function cleanImage
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% FUNCTION %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%


function filteredComponents2 = filterComponents2(I,components,WINDOW_SIZE)

    %Remove the components which are too small

%    numbOfComponents = length(components);
%    j=1;
%    while j <= numbOfComponents
%      if (components(j).Area < WINDOW_SIZE * WINDOW_SIZE)
%          components(j) = [];
%          numbOfComponents = length(components);         
%      end
%       j=j+1;
%    end
 

    areas = [components.Area];
    idx = find(areas < WINDOW_SIZE * WINDOW_SIZE);
    %idx = find(areas < WINDOW_SIZE); 
    components(idx) = [];
    
    %Remove the components which are within another component
    
    numbOfComponents = length(components);
    i=1;
    while i <= numbOfComponents
        j=1;
        while j <= numbOfComponents
            if (components(i).BoundingBox(1)< components(j).BoundingBox(1) && ...
                components(i).BoundingBox(2)< components(j).BoundingBox(2) && ...    
                components(i).BoundingBox(1)+ components(i).BoundingBox(3)> components(j).BoundingBox(1)+components(j).BoundingBox(3) && ...
                components(i).BoundingBox(2)+ components(i).BoundingBox(4)> components(j).BoundingBox(2)+components(j).BoundingBox(4))
                components(j) = [];
                numbOfComponents = length(components);
            
            else
            j=j+1;
            end
        end     
                      
    i=i+1;    
    end
    
    %Draw a bounding box around each component
    
%      figure, imshow(I);
%      for j = 1:length(components)
%         rectangle('Position',[components(j).BoundingBox],'EdgeColor','b');        
%      end
%      
    filteredComponents2 = components;
end 
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% FUNCTION %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%


function [values valuesQ anglesHist secondGradHist pinversesHist freemanCode BoundaryPix] = getHistogramOfChainCode(bw,CURVATURE_BINS);

NO_OF_DIRECTIONS=8;
%CURVATURE_BINS=11; %%No of bins to count curvature bins

%  I=imread('z:/p.jpg');
%  level = graythresh(I);
%  bw = im2bw(I,level);

%  1       2      3  
%          |  
% 8 --------------4
%          |      
% 7        6      5
%           
bw=~bw; %Invert the image so that object pixels are SET

%look for starting pixel
[rs, cs] = find(bw == 1, 1, 'first');
%trace boundry in clockwise direction from the starting pixel
boundaryPixels = bwtraceboundary(bw,[rs cs],'N');

% %for debug only
 im = zeros(size(bw));
 for i = 1:size(boundaryPixels, 1)
     im(boundaryPixels(i, 1), boundaryPixels(i, 2)) = 1;
 end
 % figure; imagesc(im); colormap(gray);



%for function codfreeman , re-arrange the boundry pixels in the format
%[x,y,x,y,x,y.....]
boundaryPixels = [boundaryPixels(:, 2), boundaryPixels(:, 1)]';
BoundaryPix=boundaryPixels;
boundaryPixels = boundaryPixels(:);
freemanCode = codfreeman(boundaryPixels,NO_OF_DIRECTIONS);

%Since effectively there are only 4 directions horizontal,
%vertical,positive slope and negative slope; so qunatize code into 4 dirs
freemanCodeQ = mod(freemanCode,4);
%Replace zeros with 4
indices = find(freemanCodeQ==0);
freemanCodeQ(indices)=4;

%Shift code right by one position to get the difference between successive
%chain code entries
freemanCodeShifted = freemanCode([end 1:end-1]);
angles = mod(abs(freemanCode - freemanCodeShifted),NO_OF_DIRECTIONS);
angles = angles+1;

anglesShifted = angles([end 1:end-1]);
secondGrad = mod(abs(angles - anglesShifted),NO_OF_DIRECTIONS);
secondGrad = secondGrad+1;

xx=linspace(1,NO_OF_DIRECTIONS,NO_OF_DIRECTIONS);
xxQ=linspace(1,NO_OF_DIRECTIONS/2,NO_OF_DIRECTIONS/2);
xxCurvature=linspace(-1.5,1.5,CURVATURE_BINS);

%figure,hist(freemanCode,xx);
values=hist(freemanCode,xx);
%values=values./sum(values);
%values=values./norm(values);

valuesQ=hist(freemanCodeQ,xxQ);
%valuesQ=valuesQ./sum(valuesQ);

%figure,hist(angles,xx);
anglesHist = hist(angles,xx);
%anglesHist=anglesHist./sum(anglesHist);

%figure,hist(secondGrad,xx);
secondGradHist = hist(secondGrad,xx);
%secondGradHist=secondGradHist./sum(secondGradHist);

pinverses=curvatureIndex(freemanCode);
pinversesHist=hist(pinverses,xxCurvature);
%pinversesHist=pinversesHist./sum(pinversesHist);

%  figure, subplot(3,1,1),bar(values);
%          subplot(3,1,2),bar(anglesHist);
%          subplot(3,1,3),bar(secondGradHist);
% figure, bar(valuesQ);
end
%end%End function getChainCode
%%

function [upper, lower]=separateUpperAndLower(im)

upper = zeros(size(im));%Prepare the upper contour
lower = zeros(size(im));%Prepare the lower contour
 
[m n]= size(im);
 
%Scan row wise from left and from right and caputre the first text pixel
%encountered
 for i=1:m
     firstColWithText = find(im(i,:),1,'first');
     lastColWithText  = find(im(i,:),1,'last');
     
     if(firstColWithText>1)
         upper(i,firstColWithText)=1;
     end

      if(lastColWithText<n)
         lower(i,lastColWithText)=1;       
      end
 end
     
%Scan column wise from top and from bottom and capture the first text pixel
%found
   for j=1:n
     firstRowWithText = find(im(:,j),1,'first');
     lastRowWithText  = find(im(:,j),1,'last');
     
     if(firstRowWithText>1)
         upper(firstRowWithText,j)=1;
     end

      if(lastRowWithText<m)
         lower(lastRowWithText,j)=1;       
      end
   end

upper = cleanImage(upper);
lower = cleanImage(lower);

% figure,imshow((upper));
% figure,imshow((lower));

%end%End function separteUpperAndLower
end
function ICleaned = cleanImage2(I)

[bw n] = bwlabel(I,8);
if n==1
    ICleaned = I; %If theres only one component, return
else
    areas = regionprops(bw,'area'); %Find the area of each component
    bigComp = find([areas.Area]==max([areas.Area])); %Find the (label of) biggest component
    idx=find(bw~=bigComp(1)); %Find the indices of all the labels other than that of bigComp
    I(idx)=0; %Remove all the other components
    ICleaned=I; 
end
%end%End function cleanImage
end
%Needed func for GlobalFE
%codfreeman
%curvature index

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% FUNCTION %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%

function [tempHist tempHist3]=getChainCodePairsHistorgram(code)

tempHist=zeros(8);
for i=1:length(code)-1
    tempHist(code(i),code(i+1))=tempHist(code(i),code(i+1))+1;       
end
   % tempHist=tempHist./sum(sum(tempHist));

    tempHist=reshape(tempHist,1,[]);
    
    
    
tempHist3 = zeros([8,8,8]);
for i=1:length(code)-2
    tempHist3(code(i),code(i+1),code(i+2))=tempHist3(code(i),code(i+1),code(i+2))+1;       
end
%tempHist3=tempHist3./sum(sum(sum(tempHist3)));
    tempHist3=reshape(tempHist3,1,[]);
end
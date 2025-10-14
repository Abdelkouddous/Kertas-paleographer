function [X] = Polygon_Features_Extraction_PRL(Fichier)

clc;


imageNamesAndLines=[];
polyTimes=[];
featureTimes=[];
SIG=0;

   I = imread(Fichier);  
   I = im2uint8(I);
   I  = padarray(I,[13 13],255);
   if(SIG)
       bw=I;
   else
       bw = im2bw(I,graythresh(I));
   end

   [ignore1 ignore2 ignore3 lines comps polyTime] = getChainCodedImageWithInteriorContours(bw);
   tic
   [lineAngles slopesHist lengthsAtNSlopes histOfLengths lengthsAtNAngles]=lineBasedFeaturesNewBins(lines,comps);
   featureTime=toc;
   polyTimes=[polyTimes polyTime];
   featureTimes=[featureTimes featureTime];
   
   X = [lineAngles slopesHist lengthsAtNSlopes histOfLengths lengthsAtNAngles];
  % save(Fichier1, 'X');  % Polygon Based Features.
   end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bwCoded bwCodedAngles bwCoded2ndGrad imageLines componentsInfo polyTime]=getChainCodedImageWithInteriorContours(bw)

VISUALIZE=0;
if(VISUALIZE)
 figure,imagesc(bw);
 hold on,
end
%Accepts a binary image and returns the image where the contours are
%replace by the chain codes
%% Initializers/CCs
imageLines=[];
componentsInfo=[];
NO_OF_DIRECTIONS=8;
bwCoded=zeros(size(bw));
bwCodedAngles=zeros(size(bw));
bwCoded2ndGrad=zeros(size(bw));
[labels n] = bwlabel(~bw);
components = regionprops(labels);
%components = filterComponents(bw,components,10);
n=length(components);
%% Process
componentNumber=0;
polyTime=0;
%For each of the components
for i=1:n
    %% Find component bounding box
    startX = components(i).BoundingBox(1)+0.5;
    startY = components(i).BoundingBox(2)+0.5;
    w = components(i).BoundingBox(3);
    h = components(i).BoundingBox(4);
    %% Extract the component
    cc = bw(startY:startY+h-1,startX:startX+w-1);

%% % freeman code    
    cc=~cc;
    codedImage= zeros(size(cc));
    codedImageAngles= zeros(size(cc));
    codedImage2ndGrad= zeros(size(cc));
    [B L]=bwboundaries(cc,8);%Find boundaries of all the objects in this component
    numOfObjects = length(unique(L));%Find how many objects are there, ZEROs represent background
    objectProps=regionprops(L);%Find the properties of these objects
    for j=1:numOfObjects-1
        if(objectProps(j).Area>15)
            indices= find(L==j);
            curObject = zeros(size(cc));
            curObject(indices) = 1;
            %look for starting pixel
            [rs, cs] = find(curObject == 1, 1, 'first');
            %trace boundry in clockwise direction from the starting pixel
            boundaryPixels = bwtraceboundary(curObject,[rs cs],'N');

            % %for debug only
            im = zeros(size(curObject));
            for i = 1:size(boundaryPixels, 1)
                im(boundaryPixels(i, 1), boundaryPixels(i, 2)) = 1;
            end
            %  figure; imagesc(im); colormap(gray);

            %for function codfreeman , re-arrange the boundry pixels in the format
            %[x,y,x,y,x,y.....]
            boundaryPixels = [boundaryPixels(:, 2), boundaryPixels(:, 1)]';
            boundaryP=boundaryPixels;
            boundaryPixels = boundaryPixels(:);
            code = codfreeman(boundaryPixels,NO_OF_DIRECTIONS);

%% This part will create images that replace the contour by 1st and 2nd
%% Derivative of chain code

codeShifted = code([end 1:end-1]);
angles = mod(abs(code - codeShifted),NO_OF_DIRECTIONS);
angles = angles+1;

anglesShifted = angles([end 1:end-1]);
secondGrad = mod(abs(angles - anglesShifted),NO_OF_DIRECTIONS);
secondGrad = secondGrad+1;
            
%% Construct the component    

if(length(boundaryP)<3)
    continue;
end
try
    for kk=2:length(boundaryP)-1
        codedImage(boundaryP(2,kk),boundaryP(1,kk))=code(kk-1);
        codedImageAngles(boundaryP(2,kk),boundaryP(1,kk))=angles(kk-1);
        codedImage2ndGrad(boundaryP(2,kk),boundaryP(1,kk))=secondGrad(kk-1);
    end

    codedImage(boundaryP(2,1),boundaryP(1,1))=code(kk);
    codedImageAngles(boundaryP(2,1),boundaryP(1,1))=angles(kk);
    codedImage2ndGrad(boundaryP(2,1),boundaryP(1,1))=secondGrad(kk);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
   lines= polygonize(code,boundaryP,size(codedImage,1),size(codedImage,2));
   polyTime=polyTime+toc;

   %To find line cordinates in the actual image
   toAdd = [startX-1 startY-1 startX-1 startY-1];
   toAdd = repmat(toAdd,size(lines,1),1);
   lines = lines+toAdd;
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    componentNumber=componentNumber+1;
    
    imageLines=[imageLines;lines];
    cInfo=repmat(componentNumber,size(lines,1),1);%Keep the component number so that we know
    %which lines belong to which component
    componentsInfo = [componentsInfo; cInfo];
    
    %This part plots the lines on the components
    if(VISUALIZE)
   xx=lines(:,[1 3]);
   xx=reshape(xx',[],1);
   xx=[xx ;xx(1)];
   
   yy=lines(:,[2 4]);
   yy=reshape(yy',[],1);
   yy=[yy; yy(1)];
   plot(xx,yy,'y');
   
   %% This part plots gray characters and black line
%    map=[0.7 0.7 0.7;1 1 1];
%    colormap(map);
%    plot(xx,yy,'k','LineWidth',1);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    bwCoded(startY:startY+h-1,startX:startX+w-1)=codedImage;
    bwCodedAngles(startY:startY+h-1,startX:startX+w-1)=codedImageAngles;
    bwCoded2ndGrad(startY:startY+h-1,startX:startX+w-1)=codedImage2ndGrad;
    
catch
    boundaryP
end
        end%End if
    end%end objects loop
end%End components loop
%imagesc(bwCoded),colormap(jet),colorbar;
%end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lineAngles slopesHist lengthsAtNSlopes histOfLengths lengthsAtNAngles]=lineBasedFeaturesNewBins(lines,comps)

%This function calculates features that are extracted from polygonized
%image of handwriting
ANGLE_BINS=9;

%lines = x1 y1 x2 y2;x1 y1 x2 y2; ....... Each row represents one line

%% SLOPE
%m=y2-y1/x2-x1
slopes = (lines(:,4)-lines(:,2))./(lines(:,3)-lines(:,1));
slopes = round(atan(slopes).* (180/pi));%Convert to degrees -90 to 90
%slopesHist = hist(slopes,linspace(-90,90,ANGLE_BINS));
slopesHist = findDistOfSlopes(slopes);



%% LENGTH
%l=d=Euclidean Distance between two points of the line
lengths = sqrt((lines(:,4)-lines(:,2)).^2 + (lines(:,3)-lines(:,1)).^2);
%usefulIdx=find(lengths>3);
lengthsAtNSlopes=findLengthsAtNSlopes(slopes,lengths);

%% HIST OF LENGTHS
histOfLengths = hist(lengths,linspace(0,100,10));
histOfLengths = histOfLengths./sum(histOfLengths);


%% ANGLES
%Represent each line by a vector, line between x1,y1 and x2,y2, vector is
%x2-x1,y2-y1
vectors = [lines(:,3)-lines(:,1) lines(:,4)-lines(:,2)];
numberOfComponents = length(unique(comps));

%Angles between to connected lines
anglesBetweenLines=[];
lengthOfRays=[];
%We first need to find all the lines that belong to a given component
for i=1:numberOfComponents
    indicesForThisComp=find(comps==i);
    vectorsForThisComp= vectors(indicesForThisComp,:);
    vectorsShifted = [vectorsForThisComp(end,:) ; vectorsForThisComp(1:end-1,:)];

    %Perform the dot product on the vector matrix of current component.
    %Since each vector is to be multiplied with the next one (line with next line)
    %Shift the matrix by one position and apply dot product
    
    %    costheta = a.b /|a||b|
    cosTheta = [vectorsShifted(:,1).*vectorsForThisComp(:,1) + vectorsShifted(:,2).*vectorsForThisComp(:,2)];
    magnitudes=(sqrt(vectorsShifted(:,1).^2+vectorsShifted(:,2).^2).*sqrt(vectorsForThisComp(:,1).^2+vectorsForThisComp(:,2).^2));
    cosTheta = cosTheta./magnitudes;
    cosTheta = round(acos(cosTheta).* (180/pi));
    anglesBetweenLines=[anglesBetweenLines;cosTheta];
    lengthOfRays =[lengthOfRays; round((sqrt(vectorsShifted(:,1).^2+vectorsShifted(:,2).^2)+sqrt(vectorsForThisComp(:,1).^2+vectorsForThisComp(:,2).^2)))];

end

%% Angles Between Segments
lineAngles = findDistOfLineAngles(anglesBetweenLines);

%% Length of Angles between Segments
lengthsAtNAngles=findLengthsAtNAngles(anglesBetweenLines,lengthOfRays);
end 

%end %End function

%% Distribution of Slopes



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function slopesHist = findDistOfSlopes(slopes)

%Non uniform division
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xx=[-90 -80 -55 -35 -10 10 35 55 80 90];
slopesHist = [];

for i=1:length(xx)-1
    lowerBound = xx(i);
    upperBound= xx(i+1);
    idx = find(slopes > lowerBound & slopes <= upperBound);
    slopesHist = [slopesHist (length(idx))];    
end

%Last and first is the same bin:vertical
slopesHist(1)=slopesHist(1)+slopesHist(end);
slopesHist=slopesHist(1:end-1);

%Normalize
slopesHist=slopesHist./sum(slopesHist);
%end%End function
end 

%% Distribution of Angles between segments

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function lineAngles = findDistOfLineAngles(anglesBetweenLines)

%USE THIS FOR REGULAR DIVISION
%lineAngles = hist(anglesBetweenLines,linspace(0,180,ANGLE_BINS));

xx=[-91 -80 -55 -35 -10 10 35 55 80 90];
%From each value subtract 90 so that the interval is -90 to 90
anglesBetweenLines = anglesBetweenLines - 90;
lineAngles =[];

for i=1:length(xx)-1
    lowerBound = xx(i);
    upperBound= xx(i+1);
    idx = find(anglesBetweenLines > lowerBound & anglesBetweenLines <= upperBound);
    lineAngles = [lineAngles (length(idx))];    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%New
%Last and first is the same bin:vertical
lineAngles(1)=lineAngles(1)+lineAngles(end);
lineAngles=lineAngles(1:end-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Normalize
lineAngles = lineAngles./sum(lineAngles);

%end %End function
end


%% Lengths at N Slopes
%Instead of finding the lenghts of the lines are four principal
%directions;this function finds the lenghts at N directions


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lengthsAtNSlopes=findLengthsAtNSlopes(slopes,lengths)

xx=[-91 -80 -55 -35 -10 10 35 55 80 90];
lengthsAtNSlopes = [];

for i=1:length(xx)-1
    lowerBound = xx(i);
    upperBound= xx(i+1);
    idx = find(slopes > lowerBound & slopes <= upperBound);
    lengthsAtNSlopes = [lengthsAtNSlopes sum(lengths(idx))];    
end
%Last and first is the same bin:vertical
lengthsAtNSlopes(1)=lengthsAtNSlopes(1)+lengthsAtNSlopes(end);
lengthsAtNSlopes=lengthsAtNSlopes(1:end-1);
%Normalize
lengthsAtNSlopes = lengthsAtNSlopes./sum(lengthsAtNSlopes);

%end%End function
end

%% Lengths at N Angles

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lengthsAtNAngles=findLengthsAtNAngles(lineAngles,lengthOfRays)

xx=[0 10 35 55 80 100 125 145 170 180];
lengthsAtNAngles=[];

for i=1:length(xx)-1
    lowerBound = xx(i);
    upperBound= xx(i+1);
    idx = find(lineAngles > lowerBound & lineAngles <= upperBound);
    lengthsAtNAngles = [lengthsAtNAngles sum(lengthOfRays(idx))];    
end
%Last and first is the same bin:vertical
lengthsAtNAngles(1)=lengthsAtNAngles(1)+lengthsAtNAngles(end);
lengthsAtNAngles=lengthsAtNAngles(1:end-1);
%Normalize
lengthsAtNAngles = lengthsAtNAngles./sum(lengthsAtNAngles);
%end%End function
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function lines=polygonize(code,boundaryPoints,YSIZE,XSIZE)

%% Arguments
%Code = chain code of the object
%boundaryPoints = contour points of object
%YSIZE/XSIZE = size of object bounding box

%%
T=2;
codeLength=size(code,1)+1;
lines=[];
%Since the origin in image is top left, we need to subtract each value from
%the max number of rows to shift the origin at the bottom left
yValues = boundaryPoints(2,:);
yValues = YSIZE+1-yValues;
boundaryPoints = [boundaryPoints(1,:);yValues];


x = boundaryPoints(1,:);
y = boundaryPoints(2,:);

subX = x(1);%Find factors to subtract
subY = y(1);


x=x-subX;%To bring the first point at origin
y=y-subY;
f(1)=0;
i=2;
start=1;
while i<=codeLength
    f(i) = f(i-1)+ (x(i)*(y(i)-y(i-1)) - y(i)*(x(i)-x(i-1)));
    L(i) = sqrt(x(i).^2 + y(i).^2);

    if (abs(f(i)) <= T*L(i))
        i=i+1;
        if(i==codeLength)
        line=[x(start)+subX YSIZE-(y(start)+subY)+1 x(i-1)+subX YSIZE-(y(i-1)+subY)+1];%Starting and end points        
        lines =[lines;line];%Add to lines collection

        end
        
    else
        %Preserve This Line
        line=[x(start)+subX YSIZE-(y(start)+subY)+1 x(i-1)+subX YSIZE-(y(i-1)+subY)+1];%Starting and end points        
        lines =[lines;line];%Add to lines collection
       
        %Put back the cordianate system so that subtractsions are not
        %accumulated
        x=x+subX; 
        y=y+subY;
        
        f(i-1)=0;%Start new line now
        start=i;%Preserve Starting point

        subX=x(i);%Update the factors to subtract
        x=x-subX; %Chose new origin
        
        subY=y(i);
        y=y-subY;
       
    end %End if

end %End while
end
%end%End function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



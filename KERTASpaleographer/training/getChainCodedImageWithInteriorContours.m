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
end
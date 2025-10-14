function windowsImage = divideWritingNaturalComponent(image,components,WINDOW_SIZE)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Limitations
% 
%   1.If skeleton exits from two different positions from the same edge
%   (E,W,N or S)
%
%   2. Keep the windows on each component separtely instead of the image so
%   that overlap could be avoided
%
%   3. In SetWindowFlags The four corner points are catered in skeleton
%   but not in labeled image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[mainImageRows mainImageCols] = size(image);
mainIndex=1;


for i= 1:length(components) % For each component

    index =1;
    windows=[];
    
    startX = components(i).BoundingBox(1)+0.5; %Convert to pixel cordinates    
    startY = components(i).BoundingBox(2)+0.5;
    
    w = components(i).BoundingBox(3);
    h = components(i).BoundingBox(4);
    
    I = image(startY:startY+h-1,startX:startX+w-1);
   
    I  = padarray(I,[WINDOW_SIZE WINDOW_SIZE],1);
    I=cleanImage(I);
   
   

    iInv = ~I;%Invert the image to find skeleton
    skeletonInv =  bwmorph(iInv,'skel',Inf);%Find image skeleton
    skeleton= ~skeletonInv; %Invert it back to have text as black

    %figure,imshow(I);
    
    [imageRows imageCols] = size(I);
    Ilabeled = ones(imageRows,imageCols);%Same size image as I,contains ZEROS in the areas where a window has been drawn
    
    stack = struct('x',0,'y',0,'remove',0,'flags',[0 0 0 0]);%Stack to contain the windows
    currentWindow = struct('x',0,'y',0,'remove',0,'flags',[0 0 0 0]);
   
   %Make the first Window
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    flag = 0;
    for x=1:imageCols
        for y = 1:imageRows
            if (I(y,x)==0)
                flag =1;
                windows(index).x=x;
                windows(index).y=moveVertical(I,Ilabeled,x,y,WINDOW_SIZE,2);
                windows(index).remove=0;
                %rectangle('Position', [windows(index).x-0.5 windows(index).y-0.5 WINDOW_SIZE WINDOW_SIZE],'EdgeColor','b');
                Ilabeled(windows(index).y:windows(index).y+WINDOW_SIZE-1,windows(index).x:windows(index).x+WINDOW_SIZE-1)=0;
                
                break;                
            end %end if
            
        end %End rows 
        if(flag)
            break;
        end
    end %End cols
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    %Set all flags of first window to zero
    windows(index).flags(1) =0;%East
    windows(index).flags(2) =0;%West
    windows(index).flags(3) =0;%North
    windows(index).flags(4) =0;%South   
    
    
    %Copy first window in lastWindow and set its flags
    lastWindow = setWindowFlags(skeleton,Ilabeled,windows(index),WINDOW_SIZE);
    lastWindow.x = windows(index).x;
    lastWindow.y = windows(index).y;
    lastWindow.remove=0;

    index=index+1;
    %Push the first window onto the stack
    stack(end) = lastWindow;

    %While stack is NOT empty
    while(length(stack) > 0)
        
        %Pop last window from Stack
        lastWindow = stack(end);


        stack(end)=[];
        windowDrawn = false;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %If only one flag is set        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if(sum(lastWindow.flags)==1)
            if(lastWindow.flags(1)) % East
                
                %Place next window on the right and adjust it vertically
                currentWindow.x = lastWindow.x + WINDOW_SIZE;
                currentWindow.y = moveVertical(I,Ilabeled,currentWindow.x,lastWindow.y,WINDOW_SIZE,2);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end
                    
                windowDrawn=true;
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(2) = 0;                        
                   
            end %End East
            
            if(lastWindow.flags(2))%West                                       
                              
                %Place next window on the left and adjust it vertically
                currentWindow.x = lastWindow.x - WINDOW_SIZE;
                currentWindow.y = moveVertical(I,Ilabeled,currentWindow.x,lastWindow.y,WINDOW_SIZE,1);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end

                windowDrawn=true;
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(1) = 0;                                                 
                                
            end%End west
            
            if(lastWindow.flags(3))%North
                
                %Place the window on top and adjust it horizontally
                currentWindow.y = lastWindow.y - WINDOW_SIZE;
                currentWindow.x = moveHorizontal(I,Ilabeled,lastWindow.x,currentWindow.y,WINDOW_SIZE,4);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end
                    
                windowDrawn=true;
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;             
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(4)=0;                                                               
                               
            end%End North
            
            if(lastWindow.flags(4))%South
                
                %Place the window below and move it horizontally
                currentWindow.y = lastWindow.y + WINDOW_SIZE;
                currentWindow.x = moveHorizontal(I,Ilabeled,lastWindow.x,currentWindow.y,WINDOW_SIZE,3);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end
                    
                windowDrawn=true;
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(3)=0;                                 
                             
            end%South
            
            %Add the new window to the list of windows
            if(windowDrawn)
                %rectangle('Position', [currentWindow.x-0.5 currentWindow.y-0.5 WINDOW_SIZE WINDOW_SIZE],'EdgeColor','r');            
                windows(index) = tempWindow;
                index = index+1;
               
                %If any flag of this window is set,push it on stack
                 if(sum(tempWindow.flags)>0)
                     stack(end+1)=tempWindow;
                 end
            end%End if window drawn
             
        end % End if one flag is set
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %If more than one flag is set        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if(sum(lastWindow.flags)>1)
            
            firstNonZeroFlag=find(lastWindow.flags,1);
            
            if(firstNonZeroFlag==1) % East

                lastWindow.flags(1)=0;
                
                %Place the window on the right and move it vertically
                currentWindow.x = lastWindow.x + WINDOW_SIZE;
                currentWindow.y = moveVertical(I,Ilabeled,currentWindow.x,lastWindow.y,WINDOW_SIZE,2);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end
                    
                windowDrawn=true;        
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(2) = 0;                                   
                              
            end %End East
            
            if(firstNonZeroFlag==2)%West
                
                lastWindow.flags(2)=0;
                
                %Place the window on left and move it vertically
                currentWindow.x = lastWindow.x - WINDOW_SIZE;
                currentWindow.y = moveVertical(I,Ilabeled,currentWindow.x,lastWindow.y,WINDOW_SIZE,1);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;    
                end

                windowDrawn=true;              
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(1) =0;                                
                
            end%End west
            
            if(firstNonZeroFlag==3)%North
                
                lastWindow.flags(3)=0;             
                
                %Place the window on top and move it horizontally
                currentWindow.y = lastWindow.y - WINDOW_SIZE;
                currentWindow.x = moveHorizontal(I,Ilabeled,lastWindow.x,currentWindow.y,WINDOW_SIZE,4);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end

                windowDrawn=true;
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(4)=0;
                
            end%End North
            
            if(firstNonZeroFlag==4)%South
                
                lastWindow.flags(4)=0;  
                
                %Place the window below and move it horizontally
                currentWindow.y = lastWindow.y + WINDOW_SIZE;
                currentWindow.x = moveHorizontal(I,Ilabeled,lastWindow.x,currentWindow.y,WINDOW_SIZE,3);
                currentWindow.remove=0;
                
                if (overlaps(Ilabeled,currentWindow,WINDOW_SIZE))
                    currentWindow.remove=1;
                end    
                
                windowDrawn=true;
                Ilabeled(currentWindow.y:currentWindow.y+WINDOW_SIZE-1,currentWindow.x:currentWindow.x+WINDOW_SIZE-1)=0;                
                tempWindow = setWindowFlags(skeleton,Ilabeled,currentWindow,WINDOW_SIZE);            
                tempWindow.flags(3)=0;                              

            end%South      
                
                if(windowDrawn)
                   % rectangle('Position', [currentWindow.x-0.5 currentWindow.y-0.5 WINDOW_SIZE WINDOW_SIZE],'EdgeColor','g');            
                    windows(index) = tempWindow;
                    index = index+1;
               
                    %Push back the lastWindow to stack (it had more than one
                    %flags true
                    stack(end+1)=lastWindow;
                
                    %Push the current window if any of its flags is set
                    if(sum(tempWindow.flags)>0)
                        stack(end+1)=tempWindow;
                    end
                    
                end%End if windowDrawn
                
                
        end % End more than one flag

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
              
    end%End while (stack is not empty)
    
    %Remove overlaping windows
    n=length(windows);
    ii=1;
    while(ii<=n)
        if(windows(ii).remove)
            windows(ii)=[];
            n=length(windows);
        else
           ii=ii+1;
        end

    end
  
    
    %Copy the windows for this component in the array windowsImage and
    %adjust the cordiantes with respect to image    


    for ii =1:length(windows)
        windowsImage(mainIndex).x = windows(ii).x-WINDOW_SIZE+(startX-1); %Add the start X of each component 
        windowsImage(mainIndex).y = windows(ii).y-WINDOW_SIZE+(startY-1); %And minus the padding (WINDOW_SIZE)
       % rectangle('Position', [windowsImage(mainIndex).x-0.5 windowsImage(mainIndex).y-0.5 WINDOW_SIZE WINDOW_SIZE],'EdgeColor','b');
        mainIndex=mainIndex+1;        
    end
   
    %Find the branches that might have been missed
    InvertedImage = ~I;
    leftOver = ~(InvertedImage & Ilabeled);
    leftOverWindows = windowiseLeftOvers(leftOver,WINDOW_SIZE);
    
     for ii=1:length(leftOverWindows)
         windowsImage(mainIndex).x = leftOverWindows(ii).x-WINDOW_SIZE+(startX-1);
         windowsImage(mainIndex).y = leftOverWindows(ii).y-WINDOW_SIZE+(startY-1);
         mainIndex=mainIndex+1;  
     end
    
   
 
end %End components

   

  %Remove noise
   n=length(windowsImage);
   ii=1;
   while(ii<=n)
       if (windowsImage(ii).x < 1 || windowsImage(ii).y < 1 || ...
           windowsImage(ii).x+WINDOW_SIZE > mainImageCols || ...
           windowsImage(ii).y + WINDOW_SIZE > mainImageRows )
           
            windowsImage(ii)=[];
            n=length(windowsImage);
       else
           ii=ii+1;
       
       end
   end

   n=length(windowsImage);
   ii=1;
   while(ii<=n)
       if(isNoise(image(windowsImage(ii).y:windowsImage(ii).y+WINDOW_SIZE-1,windowsImage(ii).x:windowsImage(ii).x+WINDOW_SIZE-1),80,80))
            windowsImage(ii)=[];
            n=length(windowsImage);
        else
           ii=ii+1;
        end

   end
%     
%     figure,imshow(image);
%     for ii =1:length(windowsImage)
%           rectangle('Position', [windowsImage(ii).x-0.5 windowsImage(ii).y-0.5 WINDOW_SIZE WINDOW_SIZE],'EdgeColor','b');
%     end
   
end %End fucntion


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               UTILITY FUNCTIONS        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Vertical
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function row = moveVertical(I,Ilabeled,x,y,WINDOW_SIZE,prevWindow)


%Make an image of the window and find the number of components in it
windowImage = ~I(y:y+WINDOW_SIZE-1,x:x+WINDOW_SIZE-1);
[labels n] = bwlabel(windowImage,8);

%If window contains only one component
if n==1


    firstRowContainsText = max(I(y,x:x+WINDOW_SIZE-1)==0);
    lastRowContainsText = max(I(y+WINDOW_SIZE-1,x:x+WINDOW_SIZE-1)==0);

    %If ONLY first row contains text, move up
    if( firstRowContainsText && ~lastRowContainsText)
        row = moveup(I,Ilabeled,x,y,WINDOW_SIZE);    
    
    %If ONLY last row contains text, move up
    else if( ~firstRowContainsText && lastRowContainsText)
        row = movedown(I,Ilabeled,x,y,WINDOW_SIZE);  
    
        %If NONE contains text, move up
        else if ( ~firstRowContainsText && ~lastRowContainsText)
            row = moveup(I,Ilabeled,x,y,WINDOW_SIZE);    
            
        %If BOTH contain text, draw window
            else 
                row = y;
            end
        end
    end
else
    
    row = adjustRow(I,Ilabeled,x,y,WINDOW_SIZE,prevWindow);
    
end %End if n==1

end        


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Horizontal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function col = moveHorizontal(I,Ilabeled,x,y,WINDOW_SIZE,prevWindow)


%Make an image of the window and find the number of components in it
windowImage = ~I(y:y+WINDOW_SIZE-1,x:x+WINDOW_SIZE-1);
[labels n] = bwlabel(windowImage,8);

%If window contains only one component
if n==1
    
    firstColContainsText = max(I(y:y+WINDOW_SIZE-1,x)==0);
    lastColContainsText = max(I(y:y+WINDOW_SIZE-1,x+WINDOW_SIZE-1)==0);

    %If ONLY first col contains text, move left
    if(firstColContainsText && ~lastColContainsText)
        col = moveleft(I,Ilabeled,x,y,WINDOW_SIZE);
    
    %If ONLY last col contains text, move right
    else if( ~firstColContainsText && lastColContainsText)
        col = moveright(I,Ilabeled,x,y,WINDOW_SIZE);        
    
        %If NONE contains text, move right
        else if ( ~firstColContainsText && ~lastColContainsText)
            col = moveright(I,Ilabeled,x,y,WINDOW_SIZE);    
            
            %If BOTH contain text, draw window
            else 
               col=x;
            end
        end
    end
else
     col = adjustCol(I,Ilabeled,x,y,WINDOW_SIZE,prevWindow);
end %End if n==1

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function row = moveup(I,Ilabeled,x,y,WINDOW_SIZE)

row =y+WINDOW_SIZE-1;
while(~ max(I(row,x:x+WINDOW_SIZE-1)==0)==1 && ...
        all(Ilabeled(row,x:x+WINDOW_SIZE-1)) && ...
        row>y                                   ...%can not move infinitely
     ) %Move up until the last row contains a black pixel
    row=row-1;   
    
end
row = row-WINDOW_SIZE+1; %Adjust so that row contains the top left corner 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Down
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function row = movedown(I,Ilabeled,x,y,WINDOW_SIZE)
row=y;
while(~ max(I(row,x:x+WINDOW_SIZE-1)==0)==1 && ...
        all(Ilabeled(row,x:x+WINDOW_SIZE-1))&&...
        row < y+WINDOW_SIZE-1 ...
     ) %Move down until the first row contains a black pixel
    row=row+1;    
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Left
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function col = moveleft(I,Ilabeled,x,y,WINDOW_SIZE)

col=x+WINDOW_SIZE-1;
while(~ max(I(y:y+WINDOW_SIZE-1,col)==0)==1 && ...
        all(Ilabeled(y:y+WINDOW_SIZE-1,col))...
     ) %Move left until the last col contains a black pixel
    col=col-1;    
end
col = col-WINDOW_SIZE+1; %Adjust so that col contains the top left corner 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Right
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function col = moveright(I,Ilabeled,x,y,WINDOW_SIZE)

col=x;
while(~ max(I(y:y+WINDOW_SIZE-1,col)==0)==1 && ...
        all(Ilabeled(y:y+WINDOW_SIZE-1,col))...
     ) %Move right until the first col contains a black pixel
    col=col+1;    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Set Window Flags
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function windowUpdated = setWindowFlags(I,Ilabeled,window,WINDOW_SIZE)

row = window.y;
col = window.x;

try
    window.flags(1) =  min( max ( I(row-1:row+WINDOW_SIZE,col+WINDOW_SIZE)==0), ...                      
                       ~max ( Ilabeled(row-1:row+WINDOW_SIZE,col+WINDOW_SIZE)==0) ...
                       ); %East, cater corner points as well
                    % ~max ( Ilabeled(row:row+WINDOW_SIZE-1,col+WINDOW_SIZE)==0) ...
catch                 
                  
    window.flags(1) = 0;   
end

try
    window.flags(2) =  min ( max (I(row-1:row+WINDOW_SIZE,col-1)==0),...
                        ~max (Ilabeled(row-1:row+WINDOW_SIZE,col-1)==0)...
                       );%West , Cater corner points as well
                       %~max (Ilabeled(row:row+WINDOW_SIZE-1,col-1)==0)...
catch
    window.flags(2)=0;
end

try
    window.flags(3)=   min ( max (I(row-1,col:col+WINDOW_SIZE-1)==0), ...
                         ~max (Ilabeled(row-1,col:col+WINDOW_SIZE-1)==0)...
                       );%North
                        
catch
    window.flags(3)=0;
end

try
    window.flags(4) =  min ( max (I(row+WINDOW_SIZE,col:col+WINDOW_SIZE-1)==0), ...
                        ~max (Ilabeled(row+WINDOW_SIZE,col:col+WINDOW_SIZE-1)==0) ...
                       );%South
catch
    window.flags(4)=0;
end


windowUpdated = window;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Adjust Row
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function row = adjustRow(I,Ilabeled,x,y,WINDOW_SIZE,prevWindow)


%Make an image of the window and find the number of components in it
windowImage = ~I(y:y+WINDOW_SIZE-1,x:x+WINDOW_SIZE-1);
[labels n] = bwlabel(windowImage,8);

%If window contains only one component
if n==1 || n==0
    row=y;
else

 if(prevWindow==2) %Previous window was on west
    
    
     [i j labelOfInterest] = find(labels(:,1)~=0,1,'first');
     labelOfInterest = labels(i,1);
           
     
 else if(prevWindow ==1) %Previous window was on east

     
     [i j labelOfInterest] = find(labels(:,end)~=0,1,'first');
     labelOfInterest = labels(i,end);
        
     end %End if prevWindow == 1 
     
 end %End if prevWindow == 2 max
 

firstRowContainsLabel = 1;
lastRowContainsLabel = 1;

if(labelOfInterest)
    firstRowContainsLabel = max(labels(1,:)==labelOfInterest);
    lastRowContainsLabel = max(labels(end,:)==labelOfInterest);
end

%If ONLY first row contains label of interest, move up
if( firstRowContainsLabel && ~lastRowContainsLabel)
    row = moveup2(labels,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest);    
    
    %If ONLY last row contains label of interest, move down
else if( ~firstRowContainsLabel && lastRowContainsLabel)
    row = movedown2(labels,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest);  
    
        %If NONE contains text, move up
    else if ( ~firstRowContainsLabel && ~lastRowContainsLabel)
            row = moveup2(labels,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest);    
            
        %If BOTH contain, draw window
        else 
            row = y;
        end
    end
end
 
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Adjust Col
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function col = adjustCol(I,Ilabeled,x,y,WINDOW_SIZE,prevWindow)


%Make an image of the window and find the number of components in it
windowImage = ~I(y:y+WINDOW_SIZE-1,x:x+WINDOW_SIZE-1);
[labels n] = bwlabel(windowImage,8);

%If window contains only one component
if n==1 || n==0
    col=x;
else
    if(prevWindow==4) %Previous window was in south          
        [i j labelOfInterest] = find(labels(end,:)~=0,1,'first'); 
        labelOfInterest = labels(end,j);
        
        
    else if (prevWindow ==3)%Previous window was in north                          
            [i j labelOfInterest] = find(labels(1,:)~=0,1,'first');
            labelOfInterest = labels(1,j);
        end  % end prevWind == 3
    end % end prevWind == 4

firstColContainsLabel = max(labels(:,1)==labelOfInterest);
lastColContainsLabel = max(labels(:,end)==labelOfInterest);

%If ONLY first col contains label of interest, move left
if( firstColContainsLabel && ~lastColContainsLabel)
    col = moveleft2(labels,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest);    
    
    %If ONLY last col contains label of interest, move right
else if( ~firstColContainsLabel && lastColContainsLabel)
    col = moveright2(labels,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest);  
    
        %If NONE contains, move right
    else if ( ~firstColContainsLabel && ~lastColContainsLabel)
            col = moveright2(labels,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest);    
            
        %If BOTH contain, draw window
        else 
            col = x;
        end
    end
end
 
    
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Up2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rowAdjusted = moveup2(I,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest)

row =y+WINDOW_SIZE-1;
rowInLabel=WINDOW_SIZE;
while(~ max(I(rowInLabel,:)==labelOfInterest)==1 && ...
        all(Ilabeled(row,x:x+WINDOW_SIZE-1)) && ...
        row > y                 ...
     ) %Move up untilthe last row contains label of interest
    row=row-1;    
    rowInLabel = rowInLabel-1;    
end
moveBy = WINDOW_SIZE-rowInLabel; %Adjust so that row contains the top left corner 
rowAdjusted = y - moveBy;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Down2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rowAdjusted = movedown2(I,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest)

row =y;
rowInLabel=1;
while(~ max(I(rowInLabel,:)==labelOfInterest)==1 && ...
        all(Ilabeled(row,x:x+WINDOW_SIZE-1)) && ...
        row  < y+WINDOW_SIZE-1 ...
     ) %Move up until the first row contains label of interest
    row=row+1;    
    rowInLabel = rowInLabel+1;    
end

rowAdjusted = y +rowInLabel;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Left2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function colAdjusted = moveleft2(I,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest)

col =x+WINDOW_SIZE-1;
colInLabel=WINDOW_SIZE;
while(~ max(I(:,colInLabel)==labelOfInterest)==1 && ...
        all(Ilabeled(y:y+WINDOW_SIZE-1,col))  ...
     ) %Move left until the last col contains label of interest
    col=col-1;    
    colInLabel = colInLabel-1;    
end
moveBy = WINDOW_SIZE-colInLabel; 
colAdjusted = x - moveBy;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Move Right2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function colAdjusted = moveright2(I,Ilabeled,x,y,WINDOW_SIZE,labelOfInterest)

col =x;
colInLabel=1;
while(~ max(I(:,colInLabel)==labelOfInterest)==1 && ...
        all(Ilabeled(y:y+WINDOW_SIZE-1,col))  ...
     ) %Move right until the first row contains label of interest
    col=col+1;    
    colInLabel = colInLabel+1;    
end

colAdjusted = x + colInLabel;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %Clean Image (to contain only one
                                %component)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ICleaned = cleanImage(I)

I=~I;%To find components, invert the image
[bw n] = bwlabel(I,8);
if n==1
    ICleaned = ~I; %If theres only one component,invert back and return
else
    areas = regionprops(bw,'area'); %Find the area of each component
    bigComp = find([areas.Area]==max([areas.Area])); %Find the (label of) biggest component
    if(numel(bigComp)>0)
    idx=find(bw~=bigComp(1)); %Find the indices of all the labels other than that of bigComp
    I(idx)=0; %Remove all the other components
    end
    ICleaned=~I; %Invert back the image    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Overlaps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = overlaps(Ilabeled,window,WINDOW_SIZE)

I = Ilabeled(window.y:window.y+WINDOW_SIZE-1,window.x:window.x+WINDOW_SIZE-1);%Pick up the data from label
numbOfZeros = sum(sum(I==0));%Count the number of pixels under this window that are already labeled
overlapPercentage = (numbOfZeros*100)/(WINDOW_SIZE*WINDOW_SIZE);
if overlapPercentage > 35
    result = true;
else
    result = false;
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                % Windowise left overs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function windows = windowiseLeftOvers(image,WINDOW_SIZE)

    
    BW = ~image; % Invert the image to be used with bwlabel
    [labeled,numOfComponents] = bwlabel(BW,8);
    components = regionprops(labeled,'basic');   
    filteredComponents = filterComponents(image,components,WINDOW_SIZE);
    if(length(filteredComponents)>0)
        windows = divideWritingNaturalComponent(image,filteredComponents,WINDOW_SIZE);
%        windows = divideWritingByComp(image,filteredComponents,WINDOW_SIZE);
    else
        windows=[];
    end
    

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                %END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






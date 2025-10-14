function filteredComponents = filterComponents(I,components,WINDOW_SIZE)



    

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
    filteredComponents = components;
    
    
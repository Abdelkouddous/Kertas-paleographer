function [distribution distribution2 distribution3] = getDirFragHistogram(bw,windows,WINDOW_SIZE,BINS)
%Accepts a binary image alongwith the windows and returns the directional
%distribution 
DIRECTIONS=8;
INTERVALS=BINS;
DIVISION_FACTOR=100/INTERVALS;
distribution = ones(DIRECTIONS,INTERVALS);%zeros default
distribution2 = ones(DIRECTIONS,INTERVALS);%zeros default
distribution3 = ones(DIRECTIONS,INTERVALS);%zeros default

lengthOfLines =zeros(1,4);
numberOfLines = zeros(1,4);

%Distribution table
%    1 |  2 |  3 |  4
% -----|----|----|-------    
% d1   |    |    |
% d2   |    |    |
% .    |    |    |
% .    |    |    |
% d8   |    |    |

%% Get the image where contours are replace by chain codes
%bwCoded = getChainCodedImage(bw);
[bwCoded bwCoded2 bwCoded3] = getChainCodedImageWithInteriorContours(bw);

%% For all the windows, find distribution
for i=1:length(windows)
 
    %Pick up the window from coded image
    imagette = bw(windows(i).y:windows(i).y+WINDOW_SIZE-1,windows(i).x:windows(i).x+WINDOW_SIZE-1); 
    imagetteCoded = bwCoded(windows(i).y:windows(i).y+WINDOW_SIZE-1,windows(i).x:windows(i).x+WINDOW_SIZE-1); 
    nonZeroIndices = find(imagetteCoded);
    codes = imagetteCoded(nonZeroIndices);
%% This part is to find lengths of horizontal vertical left and right
%% diagonal lines
% codeQ = mod(codes,4);
% indices = find(codeQ==0);
% codeQ(indices)=4;   
% modeV = mode(codeQ);
% try
% numberOfLines(modeV)=numberOfLines(modeV)+1;
% lengthOfLines(modeV)=lengthOfLines(modeV)+length(codeQ);
% catch
%     modeV
%     codeQ
% end
    
%%     %%%%%%%%%%%%%%%%THIS PART FOR LOCAL H2 AND LOCAL H3%%%%%%%%%%%%%%
    imagetteCoded = bwCoded2(windows(i).y:windows(i).y+WINDOW_SIZE-1,windows(i).x:windows(i).x+WINDOW_SIZE-1); 
    nonZeroIndices = find(imagetteCoded);
    codes2 = imagetteCoded(nonZeroIndices);
 
    imagetteCoded = bwCoded3(windows(i).y:windows(i).y+WINDOW_SIZE-1,windows(i).x:windows(i).x+WINDOW_SIZE-1); 
    nonZeroIndices = find(imagetteCoded);
    codes3 = imagetteCoded(nonZeroIndices);

    xx=linspace(1,DIRECTIONS,DIRECTIONS);
    values2=hist(codes2,xx);
    if(sum(values2)>0)
    values2=round((values2./sum(values2))*100);
    end

    values3=hist(codes3,xx);
    if(sum(values3)>0)
    values3=round((values3./sum(values3))*100);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Find histogram and hence the percentages of each direction
    xx=linspace(1,DIRECTIONS,DIRECTIONS);
    values=hist(codes,xx);
    if(sum(values)>0)
    values=round((values./sum(values))*100);
    end
    
    %Now update the table distribution by incrementing the corresponding
    %elements
    columns = ceil(values./DIVISION_FACTOR);%Find the column number
    for j=1:length(values)
        if(values(j)>0)
            distribution(j,columns(j))=distribution(j,columns(j))+1;
        end%End if
    end%end j
    
    %%%%%%%%%%%%%%%%%THIS PART FOR LOCAL H1 AND H2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    columns = ceil(values2./DIVISION_FACTOR);%Find the column number
    for j=1:length(values2)
        if(values2(j)>0)
            distribution2(j,columns(j))=distribution2(j,columns(j))+1;
        end%End if
    end%end j
    
     columns = ceil(values3./DIVISION_FACTOR);%Find the column number
    for j=1:length(values3)
        if(values3(j)>0)
            distribution3(j,columns(j))=distribution3(j,columns(j))+1;
        end%End if
    end%end j
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end%end for

%Normalize
distribution=distribution./sum(sum(distribution));
distribution2=distribution2./sum(sum(distribution2));
distribution3=distribution3./sum(sum(distribution3));

% sL = numberOfLines./sum(numberOfLines);
% sN = lengthOfLines./sum(lengthOfLines);


end%End function
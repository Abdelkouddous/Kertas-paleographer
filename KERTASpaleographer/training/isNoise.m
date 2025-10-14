function b = isNoise(I,pWhite,pBlack)

%Determines if the sub-image passed is noise or useful data

[rows, cols] = size(I);
blackPixels =0;
totalPixels = rows * cols;
for i=1:rows
    for j=1:cols
        if I(i,j) == 0
            blackPixels=blackPixels+1;
        end
    end
end

percent = (blackPixels / totalPixels)* 100;

if percent > pBlack || (100 - percent) > pWhite 
    b =true;
else
    b=false;
end





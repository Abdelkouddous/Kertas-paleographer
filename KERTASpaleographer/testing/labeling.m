%labeling function
function [Y] = labeling(Fichier)
file_name=fullfile(Fichier);
for i=1:4
Y(i)=file_name(i);
end
Y=str2num(Y);
if Y<200
    Y=1

elseif Y<300
    Y=2

elseif Y<400
    Y=3
elseif Y<500
    Y=4

elseif Y<600
    Y=5

elseif Y<700
    Y=6

elseif Y<800
    Y=7

elseif Y<900
    Y=8
elseif Y<1000
    Y=9

elseif Y<1100
    Y=10
elseif Y<1200
    Y=11
elseif Y<1300
    Y=12

elseif Y<1400
    Y=13
elseif Y<1500
    Y=14
end

end

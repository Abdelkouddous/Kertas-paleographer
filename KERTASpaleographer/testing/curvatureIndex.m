function [pinverses p2s]=curvatureIndex(freemanCode)
K=7;
if length(freemanCode)< K+1
    pinverses=[0 0 0 0 0 0 0 0 0 0 0];
    return
end
% I=imread('test.jpg');
% level=graythresh(I);
% bw=im2bw(I,level);
% [a b c freemanCode]=getHistogramOfChainCode(bw);

NO_OF_DIRECTIONS=8;
pinverses=[];
p2s=[];
freemanCode=[freemanCode(end-K+1:end); freemanCode; freemanCode(1:K)];

for i=K+1:length(freemanCode)-K
   
    backward=freemanCode(i-K:i-1); %Get K previous values
    forward=freemanCode(i+1:mod(i+K-1,length(freemanCode))+1);%Get K forward values
    xx=linspace(1,NO_OF_DIRECTIONS,NO_OF_DIRECTIONS);
    %figure,hist(backward,xx);
    %figure,hist(forward,xx);

    g=hist(backward,xx);
    f=hist(forward,xx);
    
    mg=sum(g)/NO_OF_DIRECTIONS;
    mf=sum(f)/NO_OF_DIRECTIONS;
    
    pinv=pinverse(f,mf,g,mg,NO_OF_DIRECTIONS);
    s=sign(f,g);
    p2=abs(pinv-1)*sign(f,g);
    pinverses=[pinverses pinv];
    p2s=[p2s p2];

   
end
pinverses=p2s;
end %end function

function pinv=pinverse(f,mf,g,mg,DIR)
sum=0;
sumSqrF=0;
sumSqrG=0;
for i=1:DIR
    sum=sum+(f(i)-mf)*(g(i)-mg);
    sumSqrF=sumSqrF+(f(i)-mf)*(f(i)-mf);
    sumSqrG=sumSqrG+(g(i)-mg)*(g(i)-mg);
    
end%End for
pinv=sum/(sqrt(sumSqrF*sumSqrG));
end%end function

function s=sign(f,g)
s=1;
modf=find(f==max(f));%find mode of f and g
modg=find(g==max(g));


%Shift the modes by one value to have values 0 to 7 instead of 1 to 8
modf = modf(1)-1;
modg = modg(1)-1;

%Convex Interval : [modg+1 modg+3]
convex1 = mod(modg+1,8);
convex2 = mod(modg+2,8);
convex3 = mod(modg+3,8);

concave1 = (modg-1);
if(concave1<0) concave1=8+concave1;end
concave2 = (modg-2);
if(concave2<0) concave2=8+concave2;end
concave3 = (modg-3);
if(concave3<0) concave3=8+concave3;end
convexInterval=[];
concaveInterval=[];

convexInterval = [convex1 convex2 convex3];
concaveInterval = [concave1 concave2 concave3];
if(any(convexInterval==modf) )
    s=1;
else if (any(concaveInterval==modf))
        s=-1;
    end
end
end%end function
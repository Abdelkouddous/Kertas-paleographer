function Base = codfreeman(Base, dir);
%
% CODFREEMAN
%
%   Codage de freeman d'une base image segmentée contours.
%   Chaque forme est représenté dans la base par sa liste
%   de points de contours. Le nombre de points contours
%   par forme peut être variable. Dans ce cas, la base est 
%   construite aux dimensions de la forme comportant le plus
%   grand nombre de points de contours. Les éléments vides
%   pour les formes plus petites sont positionnés à NaN.
%   (Voir CHARBASEDRAW pour plus de détails).
%
% SYNTAXE :
%
%   base = CODFREEMAN(base, dir);
%
%
% ARGUMENTS :
%
%   base 	: Base d'images sans les labels (formes rangées en colonnes).
%   dir     : nombre de directions (4 8 16)
%
% VALEURS DE RETOUR :
%
%   base    : Base codée selon le codage de freeman (4,8 ou 16 directions)
%             Les directions sont codées de 1 à dir (ex : pour dir=4, les
%             4 directions sont 1,2,3 et 4) 
%
% VOIR AUSSI :
%
%   charbasedraw
%
% COMPATIBILITE : 
%    matlab 5.3, octave 2.0
%

% Bruno Gas - LIS/P&C UPMC <gas@ccr.jussieu.fr>
% Création : 18 décembre 2000
% version : 1.2
% Derniere révision : 
% - B.Gas (22/12/2000) : debug
% - B.Gas (26/01/2001) : ~


if nargin~= 2, error('usage : Base = CODFREEMAN(Base, dir)'); end;

if dir==4   
   directions = zeros(4,2);
   directions(1:4,1) = cos((2*pi/4)*(1:4))';
   directions(1:4,2) = sin((2*pi/4)*(1:4))';      
elseif dir==8
   directions = zeros(8,2);
   directions(1:8,1) = cos((2*pi/8)*(1:8))';
   directions(1:8,2) = sin((2*pi/8)*(1:8))';      
elseif dir==16
   directions = zeros(16,2);
   directions(1:16,1) = cos((2*pi/16)*(1:16))';
   directions(1:16,2) = sin((2*pi/16)*(1:16))';      
else
   error('[CODFREEMAN] Erreur : nombre de directions : dir = {4,8 ou 16}');
   return;
end;



[MaxSymbNbr, ExNbr] = size(Base);
MaxSymbNbr = MaxSymbNbr/2;
codfr = zeros(MaxSymbNbr-1,ExNbr);
for ex=1:ExNbr
   segments = reshape(Base(:,ex),2,MaxSymbNbr)';
   [ind, ans] = find(isnan(segments(:,1))==0);
   segments = segments(ind,:);
   [SymbNbr, ans] = size(segments);
   if SymbNbr<2
      error('[CODFREEMAN] erreur: Le nombre de symboles par forme doit être supérieur à 1');               
   end;
   segments = segments(2:SymbNbr,:) - segments(1:SymbNbr-1,:);   
   normseg = sqrt(sum((segments.^2)')');
   segments = segments./(normseg*ones(1,2));  % vecteurs différence normés
   SymbNbr = SymbNbr - 1; 
   
	for i=1:SymbNbr        
      prodscal = segments(i,:)*directions';
      [ans, codfr(i,ex)] = min(prodscal);      
   end;   
end;

Base = codfr;
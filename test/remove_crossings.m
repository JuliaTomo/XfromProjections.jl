function XY = remove_crossings(XY)
%XY = REMOVE_CROSSINGS(XY)
%   Heuristics for removing corssings for closed snake
%   Author: Janus Nortoft Jensen, jnje@dtu.dk, december 2016

n = size(XY,1);
[row,col] = line_crossing_check(XY);
%XY = XY([1:end 1],:);
for k = 1:numel(row)
    i = row(k); j = col(k);
    f = i+1;
    t = j;
    if ( j-i > n/2 )
        f = j+1;
        t = i+n;
    end
    while ( f < t )
       idF = mod(f,n);
        if ( idF == 0 )
            idF = n;
        end
        f = f + 1;

        idT = mod(t,n);
        if ( idT == 0 )
            idT = n;
        end
        t = t - 1;
        tmp = XY(idF,:);
        XY(idF,:) = XY(idT,:);
        XY(idT,:) = tmp;
    end
    %XY(end,:) = XY(1,:);
end
%XY = XY(1:end-1,:);

function [row, col] = line_crossing_check(XY)
%Janus N�rtoft Jensen
%jnje@dtu.dk, December 2016
n = size(XY,1);
x = XY(:,1);
y = XY(:,2);
idx = 2:n;
idx(end+1) = 1;
xd = x(idx)-x;
yd = y(idx)-y;

a = (x-x').*yd;
b = (y-y').*xd;
c = xd.*yd';

u = (b-a)./(c-c');

h = 0<u & u<1;

intersect_matrix = h & h';
[row,col] = find(triu(intersect_matrix));

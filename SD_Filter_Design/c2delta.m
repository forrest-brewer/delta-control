function [Ad,Bd,Cd,Dd] = c2delta(A,B,C,D,ts);

Ad = (expm(A*ts)-eye(size(A)))./ts;
Bd = A\((expm(A*ts)-eye(size(A)))*B)./ts;
Cd = C;
Dd = D;

end
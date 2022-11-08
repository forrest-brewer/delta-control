function [AT,BT,CT,DT,T0] = obsv_cst(A,B,C,D)

e = zeros(size(A,2),1);
e(end) = 1;

O = obsv(A,C);

[U, S, V] = svd(O);
S_inv = S\eye(size(S));
T_inv = V*S_inv*U';
T1 = T_inv*e;

n = size(A,2);
q = size(T1,2);

T0 = zeros(n,q);

for i = 1:n
    T0(:,q*(i-1)+1:q*i) = A^(n-i)*T1;
end

AT = T0\(A*T0);
AT(:,2:size(AT,2)) = [eye(n-1);zeros(1,n-1)];
BT = T0\B;
CT = C*T0;
DT = D;

end

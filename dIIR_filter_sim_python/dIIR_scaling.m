function [Ts, k_inv] = dIIR_scaling(A,B,T0,f,ts)

T0_inv = T0\eye(size(T0));
% T0_inv = pinv(T0)*eye(size(T0));
[f_int,~] = delta_bode(A,B,T0_inv,0,f,ts);

f_norm = zeros(1,size(A,1));
for i = 1:length(f_norm)
    f_norm(i) = norm(f_int(i,:),Inf);
end

Ts = zeros(size(A));
k = zeros(1,length(f_norm));
k_inv = zeros(1,length(k));
for i = 1:length(k)
    if i == 1
        k(i) = 1/(f_norm(i));
    else
        k(i) = 1/(prod(k(1:i-1))*f_norm(i));
    end
    
    k_inv(i) = 2^floor(log2(ts/k(i)))/ts;
    %k(i) = 1/k_inv(i);
    Ts(i,i) = prod(k_inv(1:i));
    %Ts(i,i) = 1/prod(k(1:i));
end
end
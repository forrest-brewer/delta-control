function q = bitwidth_opt(S,p,H,sig2_sd,sig2_x_sd)

n = size(S,1);
u = ones(n,1);
l = zeros(n,1);
f = ones(1,n);
f(1) = 3;
sig2_x_sd(end+1) = 0;

cvx_solver sedumi
cvx_precision high
cvx_begin
    variable q(n,1) 
    maximize(f*q)
    subject to 
        quad_form(q,H) <= sig2_sd;
        S'*q <= p';
        q >= sig2_x_sd;    
cvx_end

%q = 2.^floor(log2(q))';

end
    

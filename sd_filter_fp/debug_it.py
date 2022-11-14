#
#
import numpy as np
from scipy import signal
from scipy import linalg
import control
import scipy.io as sio
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'../sdfpy')
import sdfpy as sdf

# ----------------------------------------------------------
def check_ss(bode,zz,f,ts):
  x = range(bode.shape[0])
  plt.plot(x, bode)
  mag, phz = sdf.delta_bode(zz.A,zz.B,zz.C,zz.D,f,ts)
  mag = np.squeeze(mag)
  plt.plot(x, mag)
  plt.show()
  print(linalg.norm(bode - mag))

# ----------------------------------------------------------

mat_data = sio.loadmat('SD_dIIR_sensitivity.mat')

A  = mat_data['A' ]
B  = mat_data['B' ]
C  = mat_data['C' ]
D  = mat_data['D' ]
T0 = mat_data['T0']
Ts = mat_data['Ts']
f  = mat_data['f' ]
ts = mat_data['ts']

S_bmag_mat = mat_data['S_bmag']
S_amag_mat = mat_data['S_amag']
S_mag_mat  = mat_data['S_mag' ]
S_phz_mat  = mat_data['S_phz' ]

f  = f[0]
ts = ts[0]

# function [S_mag, S_phz] = SD_dIIR_sensitivity(A,B,C,D,T0,Ts,f,ts)
[S_mag, S_phz] = sdf.SD_dIIR_sensitivity(A,B,C,D,T0,Ts,f,ts)

plt.plot(f, S_mag[0])
plt.show()
plt.plot(f, S_mag[1])
plt.plot(f, S_mag[2])
plt.plot(f, S_mag[3])
plt.plot(f, S_mag[4])
plt.show()

# print('S_mag diff', np.amax(np.abs(S_mag_mat - S_mag)))
# print('S_phz diff', np.amax(np.abs(S_phz_mat - S_phz)))


mat_data = sio.loadmat('dDFIIt_noise_gain.mat')

Ad    = mat_data['Ad' ]
Bd    = mat_data['Bd' ]
Cd    = mat_data['Cd' ]
Dd    = mat_data['Dd' ]
K_inv = mat_data['K_inv']
Ts    = mat_data['Ts']
T0    = mat_data['T0']
f     = mat_data['f' ]
ts    = mat_data['ts']

NTF_bode = mat_data['NTF_bode']
NTF_bode = np.squeeze(NTF_bode)

sig_2_sd1_mat = mat_data['sig_2_sd1']
sig_2_sd1_mat = np.squeeze(sig_2_sd1_mat)

sig_2_sd2_mat = mat_data['sig_2_sd2']
sig_2_sd2_mat = np.squeeze(sig_2_sd2_mat)

m1_mat = mat_data['m1']
m1_mat = np.squeeze(m1_mat)

H_mat = mat_data['H']
H_mat = np.squeeze(H_mat)

sig_2_x_sd2_mat = mat_data['sig_2_x_sd2']
sig_2_x_sd2_mat = np.squeeze(sig_2_x_sd2_mat)

m_sys_x_sd1_mat = mat_data['m_sys_x_sd1']
m_sys_x_sd1_mat = np.squeeze(m_sys_x_sd1_mat)

sig_2_x_sd_mat = mat_data['sig_2_x_sd']
sig_2_x_sd_mat = np.squeeze(sig_2_x_sd_mat)


ts = ts[0][0]
f  = np.squeeze(f)

# function [sig_2_nom, sig_2_x_sd, H] = dDFIIt_noise_gain(Ad,Bd,Cd,Dd,K_inv,Ts,T0,f,ts)

# % Sigma Delta Specifications (2nd Order)
# n_sd = ts/3;
n_sd = ts/3
# NTF_num = [ts^2 0 0];
# NTF_den = [ts^2 2*ts 1];
NTF_num = [ts**2, 0, 0]
NTF_den = [ts**2, 2*ts, 1]
# NTF = ss(tf(NTF_num,NTF_den));
NTF = control.tf(NTF_num, NTF_den)
print(NTF)
NTF = control.ss(control.tf(NTF_num, NTF_den))

# check_ss(NTF_bode,NTF,f,ts)

# % noise gain due to input sigma delta
# H_sd = ss(Ad,Bd,Cd,Dd);
H_sd = control.ss(Ad,Bd,Cd,Dd);
# sys_sd1 = NTF*H_sd;
sys_sd1 = control.series(NTF, H_sd)
# [g1,~] = delta_bode(sys_sd1.a,sys_sd1.b,sys_sd1.c,sys_sd1.d,f,ts);
[g1,phz] = sdf.delta_bode(sys_sd1.A,sys_sd1.B,sys_sd1.C,sys_sd1.D,f,ts)
# sig_2_sd1 = n_sd*(squeeze(g1).^2);
sig_2_sd1 = n_sd*(np.squeeze(g1)**2)

# print(linalg.norm(sig_2_sd1 - sig_2_sd1_mat))

# % noise gain due to output sigma delta
# E_sd = ss(Ts\(T0\Ad),Ts\(T0\(Ad-eye(size(Ad))))*T0*...
    # [1 zeros(1,size(Ad,2)-1)]',Cd*T0*Ts,1);
A = linalg.solve(Ts, linalg.solve(T0, Ad))
z = np.zeros(Ad.shape[1])
z[0] = 1
B = linalg.solve(Ts, linalg.solve(T0, Ad-np.eye(Ad.shape[0])))
B = B @ T0 @ z.T
C = Cd @ T0 @ Ts
D = 1
E_sd = control.ss(A, B, C, D)
# sys_sd2 = NTF*E_sd;
sys_sd2 = control.series(NTF, E_sd)
# [g2,~] = delta_bode(sys_sd2.a,sys_sd2.b,sys_sd2.c,sys_sd2.d,f,ts);
[g2,phz] = sdf.delta_bode(sys_sd2.A,sys_sd2.B,sys_sd2.C,sys_sd2.D,f,ts)
# sig_2_sd2 = n_sd*(squeeze(g2).^2);
sig_2_sd2 = n_sd*(np.squeeze(g2)**2)

# print(linalg.norm(sig_2_sd2 - sig_2_sd2_mat))
# x = range(sig_2_sd2_mat.shape[0])
# plt.plot(x, sig_2_sd2_mat)
# plt.plot(x, sig_2_sd2)
# plt.show()


# %noise gain due to scaling coefficient multiplication roundoff
# sys_g0 = ss(Ts\(T0\Ad),Ts\(T0\(Ad-eye(size(Ad))))*T0*...
    # [1 zeros(1,size(Ad,2)-1)]',Cd*T0*Ts,1);
sys_g0 = E_sd.copy()
# sys_g = ss(Ad',Cd',Ts'*T0',0);
sys_g = control.ss(Ad.T,Cd.T,Ts.T @ T0.T, 0)
# sys_k = [sys_g0; sys_g(1:end-1); 0];
sys_g.C[-1,:] = 0
sys_g.D[-1,:] = 0
sys_k = sdf.ss_concat_outputs(sys_g0, sys_g)
# [m1,~] = delta_bode(sys_k.a,sys_k.b,sys_k.c,sys_k.d,f,ts);
[m1,phz] = sdf.delta_bode(sys_k.A,sys_k.B,sys_k.C,sys_k.D,f,ts)
# H = diag(((2*ts)/3)*trapz(f,m1.^2,3));
H = np.diag(((2*ts)/3)*np.trapz(np.squeeze(m1**2),f))

# m1 = np.squeeze(m1)
# print(linalg.norm(m1 - m1_mat))
# x = range(m1_mat.shape[0])
# plt.plot(x, m1_mat)
# plt.plot(x, m1)
# plt.show()

# print(linalg.norm(H - H_mat))

# %noise gain from input sigma delta to integrators
# sys_x_sd1 = ss(Ad,Bd,K_inv*(Ts\(T0\eye(size(Ad)))),0)*NTF;
C = linalg.solve(Ts, linalg.solve(T0, np.eye(Ad.shape[0])))
C = K_inv @ C
sys_x_sd1 = control.ss(Ad,Bd,C,0)
sys_x_sd1 = control.series(NTF, sys_x_sd1)
# [m_sys_x_sd1,~] = delta_bode(sys_x_sd1.a,sys_x_sd1.b,sys_x_sd1.c,sys_x_sd1.d,f,ts);
[m_sys_x_sd1,phz] = sdf.delta_bode(sys_x_sd1.A,sys_x_sd1.B,sys_x_sd1.C,sys_x_sd1.D,f,ts)
# sig_2_x_sd1 = n_sd*(squeeze(m_sys_x_sd1).^2);
sig_2_x_sd1 = n_sd*(np.squeeze(m_sys_x_sd1)**2)

# m_sys_x_sd1 = np.squeeze(m_sys_x_sd1)
# print(linalg.norm(m_sys_x_sd1 - m_sys_x_sd1_mat))
# x = range(m_sys_x_sd1_mat.shape[1])
# plt.plot(x, m_sys_x_sd1_mat[3])
# plt.plot(x, m_sys_x_sd1[3])
# plt.show()

# %noise gain from output sigma delta to integrators
# sys_x_sd2 = ss(Ad,Ad*T0*Ts*...
    # [1 zeros(1,size(Ad,2)-1)]',K_inv*(Ts\(T0\eye(size(Ad)))),0)*NTF;
z = np.zeros(Ad.shape[1])
z[0] = 1
B = Ad @ T0 @ Ts @ z.T
C = linalg.solve(Ts, linalg.solve(T0, np.eye(Ad.shape[0])))
C = K_inv @ C
sys_x_sd2 = control.series(NTF, control.ss(Ad, B, C, 0))
# [m_sys_x_sd2,~] = delta_bode(sys_x_sd2.a,sys_x_sd2.b,sys_x_sd2.c,sys_x_sd2.d,f,ts);
[m_sys_x_sd2,phz] = sdf.delta_bode(sys_x_sd2.A,sys_x_sd2.B,sys_x_sd2.C,sys_x_sd2.D,f,ts)
# sig_2_x_sd2 = n_sd*(squeeze(m_sys_x_sd2).^2);
sig_2_x_sd2 = n_sd*(np.squeeze(m_sys_x_sd2)**2)

# print(linalg.norm(sig_2_x_sd2 - sig_2_x_sd2_mat))


# % total SD output noise 
# sig_2_nom = trapz(f,squeeze(sig_2_sd1),1) + trapz(f,squeeze(sig_2_sd2),1);
sig_2_nom = np.trapz(np.squeeze(sig_2_sd1),f) + np.trapz(np.squeeze(sig_2_sd2),f)

# % total SD integrator noise
# sig_2_x_sd = sig_2_x_sd1 + sig_2_x_sd2;
sig_2_x_sd = sig_2_x_sd1 + sig_2_x_sd2

print(linalg.norm(sig_2_x_sd - sig_2_x_sd_mat))

# end


# function [sig_2_nom, sig_2_x_sd, H] = dDFIIt_noise_gain(Ad,Bd,Cd,Dd,K_inv,Ts,T0,f,ts)
# [sig_nom, sig_2_x_sd, h1] = dDFIIt_noise_gain(Ad,Bd,Cd,Dd,K_inv,Ts,T0,f,ts);
sig_nom = sig_2_nom
# h1 = H

OSR = 256  # %oversample ratio
# % Bitwidth Optimization
# SNR = 90;
SNR = 90
# sig_noise = 10^(-(SNR/10)); %-sig_nom;
sig_noise = 10**(-(SNR/10)); # %-sig_nom;
# p = .1*ones(1,length(S_mag));
p = .1*np.ones((1,S_mag.shape[1]))
# s = sqrt(trapz(sig_2_x_sd,2)*(12*OSR));
s = np.sqrt(np.trapz(sig_2_x_sd)*(12*OSR));

# q = bitwidth_opt(squeeze(S_mag),p,h1,sig_noise,s);

import cvxpy as cp

S_mag = np.squeeze(S_mag)
S = S_mag
sig2_sd = sig_noise
sig2_x_sd = s

# function q = bitwidth_opt(S,p,H,sig2_sd,sig2_x_sd)

# n = size(S,1);
n = S.shape[0]
# u = ones(n,1);
u = np.ones((n,1))
# l = zeros(n,1);
l = np.zeros((n,1))
# f = ones(1,n);
f = np.ones(n)
# f(1) = 3;
f[0] = 3
# sig2_x_sd(end+1) = 0;
sig2_x_sd = np.append(sig2_x_sd,0).reshape(5,1)

# cvx_solver sedumi
# cvx_precision high
# cvx_begin
    # variable q(n,1) 
    # maximize(f*q)
    # subject to 
        # quad_form(q,H) <= sig2_sd;
        # S'*q <= p';
        # q >= sig2_x_sd;    
# cvx_end

# %q = 2.^floor(log2(q))';

# end

# Define and solve the CVXPY problem.
q = cp.Variable((n,1))
prob = cp.Problem( cp.Maximize(f @ q),
                   [ cp.quad_form(q,H) <= sig2_sd
                   , S.T @ q <= p.T
                   , q >= sig2_x_sd
                   ]
                 )
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(q.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)


mat_data = sio.loadmat('sd_filter_top.mat')

ts    = mat_data['ts']
alpha = mat_data['alpha']
beta  = mat_data['beta']
k     = mat_data['k']


# qs = log2(q);
qs = np.log2(q.value)
# qs(qs>0) = 0;
qs[qs>0] = 0
# qs = ceil(abs(qs))';
qs = np.ceil(np.abs(qs)).T
# bi = ceil(log2(max(abs(alpha),abs(beta)))) + 1;
bi = np.ceil(np.log2(np.maximum(np.abs(alpha),np.abs(beta)))) + 1
# bw = bi + qs + 1;
# b_frac = qs;
bw = bi + qs + 1
b_frac = qs

# # % Simulink Model Bitwidth Parameters
# shift = round(abs(log2(ts*k)));
shift = np.round(np.abs(np.log2(ts*k)))
# # % bw_accum = 1 + ceil(abs(log2(ts))) + b_frac(2:end);
# bw_accum = 1 + ceil(abs(log2(ts))) + b_frac;
bw_accum = 1 + np.ceil(np.abs(np.log2(ts))) + b_frac
# disp(['q = ' num2str(log2(q'))]);
print('q = \n', np.log2(q.value.T))
# disp(['Coefficient bitwidths = ' num2str(bw)]);
print('Coefficient bitwidths = \n', bw)
# # % disp(['Bits of state = ' num2str(3*bw_accum(1) + sum(bw_accum(2:end)))]);


# # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# k_ts = k .* ts;
k_ts = k * ts
# k_ts(end+1) = 0;
k_ts = np.append(k_ts, 0)
# shift(end+1) = 1;
shift = np.append(shift, 1)

# for i = 1:length(alpha)
  # bitwidths(i).accum = fi(0       , 1, bw_accum(i) , b_frac(i));
  # bitwidths(i).alpha = fi(alpha(i), 1, bw(i)       , qs(i)    );
  # bitwidths(i).beta  = fi(beta(i) , 1, bw(i)       , qs(i)    );
  # bitwidths(i).k_ts  = fi(k_ts(i) , 1, shift(i) + 1, shift(i) );
# end

for idx, val in enumerate(alpha[0]):
  print('accum[', idx, '] |' , bw_accum[0,idx] , b_frac[0,idx])
  print('alpha[', idx, '] |' , bw[0,idx]       , qs[0,idx]    )
  print('beta [', idx, '] |' , bw[0,idx]       , qs[0,idx]    )
  print('k_ts [', idx, '] |' , shift[idx] + 1  , shift[idx]   )
  print('-' * 40)

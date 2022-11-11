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



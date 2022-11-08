#
#
import numpy as np
from scipy import signal
from scipy import linalg
import control
import scipy.io as sio
import sys

sys.path.insert(0,'../sdfpy')
import sdfpy as sdf


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

print('S_mag diff', np.amax(np.abs(S_mag_mat - S_mag)))
print('S_phz diff', np.amax(np.abs(S_phz_mat - S_phz)))


mat_data = sio.loadmat('dDFIIt_noise_gain.mat')

Ad    = mat_data['A' ]
Bd    = mat_data['B' ]
Cd    = mat_data['C' ]
Dd    = mat_data['D' ]
K_inv = mat_data['K_inv']
Ts    = mat_data['Ts']
T0    = mat_data['T0']
f     = mat_data['f' ]
ts    = mat_data['ts']


# function [sig_2_nom, sig_2_x_sd, H] = dDFIIt_noise_gain(Ad,Bd,Cd,Dd,K_inv,Ts,T0,f,ts)

# % Sigma Delta Specifications (2nd Order)
# n_sd = ts/3;
# NTF_num = [ts^2 0 0];
# NTF_den = [ts^2 2*ts 1];
# NTF = ss(tf(NTF_num,NTF_den));

# % noise gain due to input sigma delta
# H_sd = ss(Ad,Bd,Cd,Dd);
# sys_sd1 = NTF*H_sd;
# [g1,~] = delta_bode(sys_sd1.a,sys_sd1.b,sys_sd1.c,sys_sd1.d,f,ts);
# sig_2_sd1 = n_sd*(squeeze(g1).^2);

# % noise gain due to output sigma delta
# E_sd = ss(Ts\(T0\Ad),Ts\(T0\(Ad-eye(size(Ad))))*T0*...
    # [1 zeros(1,size(Ad,2)-1)]',Cd*T0*Ts,1);
# sys_sd2 = NTF*E_sd;
# [g2,~] = delta_bode(sys_sd2.a,sys_sd2.b,sys_sd2.c,sys_sd2.d,f,ts);
# sig_2_sd2 = n_sd*(squeeze(g2).^2);

# %noise gain due to scaling coefficient multiplication roundoff
# sys_g0 = ss(Ts\(T0\Ad),Ts\(T0\(Ad-eye(size(Ad))))*T0*...
    # [1 zeros(1,size(Ad,2)-1)]',Cd*T0*Ts,1);
# sys_g = ss(Ad',Cd',Ts'*T0',0);
# sys_k = [sys_g0; sys_g(1:end-1); 0];
# [m1,~] = delta_bode(sys_k.a,sys_k.b,sys_k.c,sys_k.d,f,ts);
# H = diag(((2*ts)/3)*trapz(f,m1.^2,3));

# %noise gain from input sigma delta to integrators
# sys_x_sd1 = ss(Ad,Bd,K_inv*(Ts\(T0\eye(size(Ad)))),0)*NTF;
# [m_sys_x_sd1,~] = delta_bode(sys_x_sd1.a,sys_x_sd1.b,sys_x_sd1.c,sys_x_sd1.d,f,ts);
# sig_2_x_sd1 = n_sd*(squeeze(m_sys_x_sd1).^2);

# %noise gain from output sigma delta to integrators
# sys_x_sd2 = ss(Ad,Ad*T0*Ts*...
    # [1 zeros(1,size(Ad,2)-1)]',K_inv*(Ts\(T0\eye(size(Ad)))),0)*NTF;
# [m_sys_x_sd2,~] = delta_bode(sys_x_sd2.a,sys_x_sd2.b,sys_x_sd2.c,sys_x_sd2.d,f,ts);
# sig_2_x_sd2 = n_sd*(squeeze(m_sys_x_sd2).^2);

# % total SD output noise 
# sig_2_nom = trapz(f,squeeze(sig_2_sd1),1) + trapz(f,squeeze(sig_2_sd2),1);

# % total SD integrator noise
# sig_2_x_sd = sig_2_x_sd1 + sig_2_x_sd2;

# end





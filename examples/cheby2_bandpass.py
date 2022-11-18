#
# Filter example - cheby2 bandpass
#
import numpy as np
from scipy import signal
from scipy import linalg
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import sys

# sys.path.insert(0,'../sdfpy')
import sdfpy as sdf

# ----------------------------------------------------------
# Filter Specifications
OSR = 256      # oversample ratio
fb = 22050     # nyquist
fs = OSR*2*fb  # sampling frequency
ts = 1/fs      # sampling period
f  = np.logspace(0,np.log10(fb),2**10)

# ----------------------------------------------------------
# Bandpass Filter - 300Hz to 3kHz
Rs = 60
Wn = 2*np.pi*np.array([300, 3000])
ftype = 'bandpass'
N = 4

[z,p,k] = signal.cheby2(N/2,Rs,Wn,ftype, analog=True, output='zpk')
[A,B,C,D] = signal.zpk2ss(z,p,k)

# ----------------------------------------------------------
[A, T] = linalg.matrix_balance(A)
B = linalg.solve(T, B) 
C = C @ T

# ----------------------------------------------------------
# Converting from Continuous Time to Sampled Time
[Ad,Bd,Cd,Dd] = sdf.c2delta(A,B,C,D,ts)

# ----------------------------------------------------------
# Structural Transformation of Filter
[Ad_t,Bd_t,Cd_t,Dd_t,T0] = sdf.obsv_cst(Ad,Bd,Cd,Dd)
[num_t, den_t] = signal.ss2tf(Ad_t,Bd_t,Cd_t,Dd_t)

# ----------------------------------------------------------
# % Scaling
# [Ts, k] = dIIR_scaling(Ad,Bd,T0,f,ts);
# def dIIR_scaling(A,B,T0,f,ts):
[Ts, k] = sdf.dIIR_scaling(Ad,Bd,T0,f,ts)

K_inv = np.diag(k);
# num_ts = [num_t(1) num_t(2:end)./(diag(Ts)')];
num_ts = num_t.copy()
num_ts[0,1:] /= np.diag(Ts).T
# den_ts = [1 den_t(2:end)./(diag(Ts)')];
den_ts = den_t.copy()
den_ts[1:] /= np.diag(Ts).T
den_ts[0] = 1

beta  = num_ts[0];
alpha = den_ts;

# ----------------------------------------------------------
# % Calculation of sensitivity matrix
[S_mag, S_phz] = sdf.SD_dIIR_sensitivity(Ad,Bd,Cd,Dd,T0,Ts,f,ts)

# % Calculation of quantization noise
[sig_nom, sig_2_x_sd, h1] = sdf.dDFIIt_noise_gain(Ad,Bd,Cd,Dd,K_inv,Ts,T0,f,ts)

SNR = 90
sig_noise = 10**(-(SNR/10))
p = .1*np.ones((1,S_mag.shape[1]))
s = np.sqrt(np.trapz(sig_2_x_sd)*(12*OSR));
S_mag = np.squeeze(S_mag)

S = S_mag
H = h1
sig2_sd = sig_noise
sig2_x_sd = s

q = sdf.bitwidth_opt(S_mag,p,h1,sig_noise,s)

# --------------------------------------------------------------------------
mdic = { 'beta'  : beta
       , 'alpha' : alpha
       , 'k'     : k
       , 'q'     : q.value
       , 'label' : 'experiment'
       }
sio.savemat("./cheby2_bandpass.mat", mdic)


# ----------------------------------------------------------
qs = np.log2(q.value)
qs[qs>0] = 0
qs = np.ceil(np.abs(qs)).T
bi = np.ceil(np.log2(np.maximum(np.abs(alpha),np.abs(beta)))) + 1
bw = bi + qs + 1
b_frac = qs

# ----------------------------------------------------------
# # % Simulink Model Bitwidth Parameters
shift = np.round(np.abs(np.log2(ts*k)))
bw_accum = 1 + np.ceil(np.abs(np.log2(ts))) + b_frac
print('q = \n', np.log2(q.value.T))
print('Coefficient bitwidths = \n', bw)


# # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k_ts = k * ts
k_ts = np.append(k_ts, 0)
shift = np.append(shift, 1)

for idx, val in enumerate(alpha):
  print('accum[', idx, '] |' , bw_accum[0,idx] , b_frac[0,idx])
  print('alpha[', idx, '] |' , bw[0,idx]       , qs[0,idx]    )
  print('beta [', idx, '] |' , bw[0,idx]       , qs[0,idx]    )
  print('k_ts [', idx, '] |' , shift[idx] + 1  , shift[idx]   )
  print('-' * 40)



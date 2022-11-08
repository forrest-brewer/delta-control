#
#
import numpy as np
from scipy import signal
from scipy import linalg
import scipy.io as sio
import sys

sys.path.insert(0,'../sdfpy')
import sdfpy as sdf

# --------------------------------------------------------------------------
# Filter Specifications
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256       # oversample ratio
fb  = 22050     # nyquist
fs  = OSR*2*fb  # sampling frequency
ts  = 1/fs      # sampling period
f   = np.logspace(0,np.log10(fb),2**10)

num_samples = 1e7  # number of simulation samples
t_stop = ts*(num_samples-1)
t = np.arange(0, t_stop, ts)

# Filter Input Signal
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
amp = 0.5
in_data = amp*2*np.random.rand(1,t.shape[0])-1 # white noise input

# Filter Design
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# cheby2 bandpass filter
Rs = 60
Wn = 2*np.pi*np.array([300, 3000])
ftype = 'bandpass'
N = 4

# filter from zeros & poles to state space
[z,p,k]   = signal.cheby2(N/2,Rs,Wn,ftype, analog=True, output='zpk')
[A,B,C,D] = signal.zpk2ss(z,p,k)

# balancing to avoid ill conditioned matrices
[A, T] = linalg.matrix_balance(A)
B = linalg.solve(T, B)
C = C @ T

# transformation to delta domain
Ad,Bd,Cd,Dd = sdf.c2delta(A,B,C,D,ts)

# % Filter Implementation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# % Structural transformation of filter
[Ad_t,Bd_t,Cd_t,Dd_t,T0] = sdf.obsv_cst(Ad,Bd,Cd,Dd)
[num_t, den_t] = signal.ss2tf(Ad_t,Bd_t,Cd_t,Dd_t)

# % Scaling node gains as right shift operations
[Ts, k] = sdf.dIIR_scaling(Ad,Bd,T0,f,ts)

num_ts      = np.copy(num_t[0])
num_ts[1:] /= np.diag(Ts)
num_ts[0]   = num_t[0][0]
den_ts      = np.copy(den_t)
den_ts[1:] /= np.diag(Ts)
den_ts[0]   = 1

beta  = num_ts;
alpha = den_ts;

# --------------------------------------------------------------------------
mdic = { 'Ad'    : Ad
       , 'Bd'    : Bd
       , 'Cd'    : Cd
       , 'Dd'    : Dd
       , 'Ad_t'  : Ad_t
       , 'Bd_t'  : Bd_t
       , 'Cd_t'  : Cd_t
       , 'Dd_t'  : Dd_t
       , 'T0'    : T0
       , 'num_t' : num_t
       , 'den_t' : den_t
       , 'Ts'    : Ts
       , 'k'     : k
       , 'beta'  : beta
       , 'alpha' : alpha
       , 'label' : 'experiment'
       }
sio.savemat("./py_to_mat.mat", mdic)

print('done!!!')

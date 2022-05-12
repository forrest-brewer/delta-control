#
#
import numpy as np
from scipy import signal
from scipy import linalg
import scipy.io as sio
# import spectrum
import c2delta
import obsv_cst

# --------------------------------------------------------------------------
def get_matlab_array(dir, name):
  dict = sio.loadmat(dir + name + '.mat')
  return dict[name]

# --------------------------------------------------------------------------
# Filter Specifications
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256                          # oversample ratio
fb = 22050                         # nyquist
fs = OSR*2*fb                      # sampling frequency
ts = 1/fs                          # sampling period
f = np.logspace(0,np.log10(fb),2**10)

# num_samples = 2^ceil(log2(1e6));
num_samples = 1e7    # number of simulation samples
t_stop = ts*(num_samples-1)
t = np.arange(0, t_stop, ts)

# Filter Input Signal
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
amp = 0.5
# fsig = 100*fs/num_samples
# in = amp*sin(2*pi*fsig*t)
in_data = amp*2*np.random.rand(1,t.shape[0])-1 # white noise input

# Filter Design
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# cheby2 bandpass filter
Rs = 60
Wn = 2*np.pi*np.array([300, 3000])
ftype = 'bandpass'
N = 4

[z,p,k] = signal.cheby2(N/2,Rs,Wn,ftype, analog=True, output='zpk')
# [A,B,C,D] = spectrum.zpk2ss(z,p,k)
[A,B,C,D] = signal.zpk2ss(z,p,k)

[A, T] = linalg.matrix_balance(A)
# B = T\B;
# B = np.linalg.inv(T) @ B
B = linalg.solve(T, B) 
# C = C*T
C = C @ T

Ad,Bd,Cd,Dd = c2delta.c2delta(A,B,C,D,ts)



# # % Filter Implementation
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# % Structural transformation of filter
[Ad_t,Bd_t,Cd_t,Dd_t,T0] = obsv_cst.obsv_cst(Ad,Bd,Cd,Dd);
[num_t, den_t] = signal.ss2tf(Ad_t,Bd_t,Cd_t,Dd_t);


mdic = { 'Ad'   : Ad
       , 'Bd'   : Bd
       , 'Cd'   : Cd
       , 'Dd'   : Dd
       , 'Ad_t' : Ad_t
       , 'Bd_t' : Bd_t
       , 'Cd_t' : Cd_t
       , 'Dd_t' : Dd_t
       , 'T0'   : T0
       , 'num_t': num_t
       , 'den_t': den_t
       , 'label': 'experiment'
       }
sio.savemat("./py_to_mat/ss.mat", mdic)

print('done!!!')


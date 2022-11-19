#
# Filter example - cheby1 bandpass
#
import numpy as np
from scipy import signal
import sys

if '../sdfpy' not in sys.path:
  sys.path.insert(0,'../sdfpy')

import sd_sim
import sdfpy as sdf

# ----------------------------------------------------------
# Filter Specifications
OSR = 256      # oversample ratio
fb  = 22050    # nyquist
fs  = OSR*2*fb # sampling frequency
ts  = 1/fs     # sampling period

# ----------------------------------------------------------
Rp    = 0.1;
Wn    = 2*np.pi*np.array([100, 500])
ftype = 'bandpass';
N     = 6;

[z,p,k]   = signal.cheby1(N/2,Rp,Wn,ftype, analog=True, output='zpk')
[A,B,C,D] = signal.zpk2ss(z,p,k)

print(A.shape, B.shape, C.shape, D.shape)

filter = sdf.sd_filter(OSR,fb)
filter.run(A,B,C,D)
sd_sim.sim_filter(filter)

# --------------------------------------------------------------------------
import scipy.io as sio

mdic = { 'beta'  : filter.beta
       , 'alpha' : filter.alpha
       , 'k'     : filter.k
       , 'q'     : filter.q.value
       }
sio.savemat("./cheby1_6th_bandpass.mat", mdic)

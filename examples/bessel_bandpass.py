#
# Filter example - bessel bandpass
#
import numpy as np
from scipy import signal
import sys

if '../sdfpy' not in sys.path:
  sys.path.insert(0,'../sdfpy')

import sdfpy as sdf
import sd_sim

# ----------------------------------------------------------
# Filter Specifications
OSR = 256      # oversample ratio
fb  = 22050    # nyquist
fs  = OSR*2*fb # sampling frequency
ts  = 1/fs     # sampling period

# ----------------------------------------------------------
# Bandpass Filter - 300Hz to 3kHz
Wn = 2*np.pi*np.array([300, 3000])
ftype = 'bandpass'
N = 4

[z,p,k]   = signal.bessel(int(N/2),Wn,ftype, analog=True, output='zpk')
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
sio.savemat("./bessel_bandpass.mat", mdic)

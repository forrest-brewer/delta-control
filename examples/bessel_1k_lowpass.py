#
# Filter example - bessel low pass
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
# lowpass filter - 1kHz
Wn    = 2*np.pi*1000
ftype = 'lowpass'
N     = 4

[z,p,k]   = signal.bessel(N, Wn, ftype, analog=True, output='zpk')
[A,B,C,D] = signal.zpk2ss(z, p, k)

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
sio.savemat("./bessel_1k_lowpass.mat", mdic)

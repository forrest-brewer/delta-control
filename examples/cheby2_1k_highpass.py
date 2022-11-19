#
# Filter example - cheby2 highpass
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
# highpass filter - 1kHz
Rs    = 60
Wn    = 2*np.pi*1000
ftype = 'highpass'
N     = 4

[z,p,k]   = signal.cheby2(N, Rs, Wn, ftype, analog=True, output='zpk')
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
sio.savemat("./cheby2_1k_highpass.mat", mdic)

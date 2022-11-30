#
# Filter example - cheby2 bandpass
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
# Bandpass Filter - 300Hz to 3kHz
Rs = 60
Wn = 2*np.pi*np.array([300, 3000])
ftype = 'bandpass'
N = 4

[z,p,k]   = signal.cheby2(N/2,Rs,Wn,ftype, analog=True, output='zpk')
[A,B,C,D] = signal.zpk2ss(z,p,k)

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
sio.savemat("./cheby2_bandpass.mat", mdic)

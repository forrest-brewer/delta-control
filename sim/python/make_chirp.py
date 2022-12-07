#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import util

# ----------------------------------------------------------
OSR = 256
fb = 22050 # nyquist
fs = fb*2
fs_to_ds = fs*OSR # sampling rate

# ----------------------------------------------------------
t = np.arange(0, 0.01, 1.0/fs_to_ds)
u = signal.chirp(t, 0, 0.005, fb/2)
u *= (2**15) - 1

# ----------------------------------------------------------
util.psd_plot(u, fs_to_ds)

# ----------------------------------------------------------
with open('chirp_hex.txt', 'w') as f:
  for x in u:
    f.write(util.int2hex(x, 16) + '\n')

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
t = np.arange(0, 1.0, 1.0/fs_to_ds)
u = np.random.rand(t.shape[0])
u -= 0.5
u *= 2.0
u *= (2**8) - 1

# ----------------------------------------------------------
util.psd_plot(u, fs_to_ds)

# ----------------------------------------------------------
with open('noise_hex.txt', 'w') as f:
  for x in u:
    f.write(util.int2hex(x, 16) + '\n')

#
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
os.environ['DISPLAY'] = ':0'
import deltasigma as ds
import pickle

# ----------------------------------------------------------
SineAmp  = 0.4
SineFreq = 0.5e3

SincOrder = 8
DecFact   = 256

T = 0.125 # Input signal duration in seconds.
fb = 22050 # nyquist
# FsOut = 8192 # set to ensure compatibility.
FsOut = 22050*2
Fs = FsOut*DecFact #Fs

# ----------------------------------------------------------
file = open('ds_tones.pickle', 'rb')
v = pickle.load(file)
file.close()

# ----------------------------------------------------------
w = ds.sinc_decimate(v, SincOrder, DecFact)

# ----------------------------------------------------------
f, Pxx_den = signal.periodogram(w, FsOut)
plt.semilogy(f, Pxx_den)
# plt.ylim([1e-7, 1e2])
plt.xlim([10, 2000])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#
import numpy as np
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
N = max(w.shape)
Nfft = min(N, 16*8192)
n = np.arange((N - Nfft)/2, (N + Nfft)/2).astype(np.int32)
W = np.fft.fft(w[n] * ds.ds_hann(Nfft)) / (Nfft / 4)
inBin = np.round(SineFreq*Nfft)/FsOut
plt.ylabel('dB')
plt.semilogx(np.arange(Nfft)/Nfft*FsOut, ds.dbv(W), label='Output signal')
f, Wp = ds.logsmooth(W, inBin)
plt.semilogx(f*FsOut, Wp, '#556B2F', linewidth=2.5)
plt.xlim([10, FsOut/2])
plt.show()

# ----------------------------------------------------------
n = np.arange(N)
n = n.astype(np.int32)
t = np.arange(max(n.shape))
plt.plot(t, w[n], 'r')
plt.ylabel('$w(t)$')
plt.xlabel('Sample #')
plt.axis([0, max(n)-min(n), -1.1, 1.1])
plt.show()

# ----------------------------------------------------------
N = max(w.shape)
U = np.fft.fft(w)/(N/4)
f = np.linspace(0, FsOut, N + 1)
f = f[:int(N/2 + 1)]
plt.semilogx(f, ds.dbv(U[:int(N/2) + 1]))
plt.xlabel('f [Hz]')
plt.ylabel('U(f) [dB]')
plt.show()


#
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['DISPLAY'] = ':0'
import deltasigma as ds
import pickle

# ----------------------------------------------------------
OSR = 256     # oversample ratio
fb = 22050    # nyquist
fs = OSR*2*fb # sampling frequency
ts = 1/fs     # sampling period

# ----------------------------------------------------------
SineAmp  = 0.4
# SineFreq = 0.5e3

SincOrder = 8
DecFact   = 256

T = 0.125 # Input signal duration in seconds.
# fb = 22050 # nyquist
# FsOut = 8192 # set to ensure compatibility.
FsOut = 22050*2
Fs = FsOut*DecFact #Fs

# ----------------------------------------------------------
file = open('delta_sigma_cof.pickle', 'rb')
delta_sigma_cof = pickle.load(file)
file.close()

# ----------------------------------------------------------
# print(delta_sigma_cof)
alpha = delta_sigma_cof['alpha'][0]
beta  = delta_sigma_cof['beta' ][0]
k     = delta_sigma_cof['k'    ][0]

# ----------------------------------------------------------
file  = open('ds_tones.pickle', 'rb')
ds_in = pickle.load(file)
file.close()

# ----------------------------------------------------------
def do_filter(x, state):
  next_state = np.zeros(4)

  y  =  beta[0] * x
  y += state[0] * k[0] * ts
  
  if y > 0:
    y = 1.0
  else:
    y = -1.0

  next_state[0]  =  beta[1] * x
  next_state[0] -= alpha[1] * y
  next_state[0] += state[1] * k[1] * ts
  next_state[0] += state[0]

  next_state[1]  =  beta[2] * x
  next_state[1] -= alpha[2] * y
  next_state[1] += state[2] * k[2] * ts
  next_state[1] += state[1]

  next_state[2]  =  beta[3] * x
  next_state[2] -= alpha[3] * y
  next_state[2] += state[3] * k[3] * ts
  next_state[2] += state[1]

  next_state[3]  =  beta[4] * x
  next_state[3] -= alpha[4] * y
  next_state[3] += state[3]

  return y, next_state

# ----------------------------------------------------------
state = np.zeros(4)
v = np.zeros(ds_in.shape[0])

# for x in v[:20]:
  # state = do_it(x, state)

for idx, x in enumerate(ds_in):
  v[idx], state = do_filter(x, state)

plt.plot(np.arange(100), v[-100:])
plt.show()

# ----------------------------------------------------------
w = ds.sinc_decimate(v, SincOrder, DecFact)

# ----------------------------------------------------------
N = max(w.shape)
Nfft = min(N, 16*8192)
n = np.arange((N - Nfft)/2, (N + Nfft)/2).astype(np.int32)
W = np.fft.fft(w[n] * ds.ds_hann(Nfft)) / (Nfft / 4)
# inBin = np.round(SineFreq*Nfft)/FsOut
plt.ylabel('dB')
plt.semilogx(np.arange(Nfft)/Nfft*FsOut, ds.dbv(W), label='Output signal')
# f, Wp = ds.logsmooth(W, inBin)
# plt.semilogx(f*FsOut, Wp, '#556B2F', linewidth=2.5)
# plt.xlim([10, FsOut/2])
plt.show()

# ----------------------------------------------------------
n = np.arange(N)
n = n.astype(np.int32)
t = np.arange(max(n.shape))
plt.plot(t, w[n], 'r')
plt.ylabel('$w(t)$')
plt.xlabel('Sample #')
# plt.axis([0, max(n)-min(n), -1.1, 1.1])
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



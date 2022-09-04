#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
os.environ['DISPLAY'] = ':0'
import deltasigma as ds
import pickle

# ----------------------------------------------------------
OSR = 256
fb = 22050 # nyquist
fs = fb*2
fs_to_ds = fs*OSR # sampling rate
ts = 1/fs_to_ds   # sampling period

# ----------------------------------------------------------
file = open('delta_sigma_cof.pickle', 'rb')
delta_sigma_cof = pickle.load(file)
file.close()

# ----------------------------------------------------------
print(delta_sigma_cof)
alpha = delta_sigma_cof['alpha'][0]
beta  = delta_sigma_cof['beta' ][0]
k     = delta_sigma_cof['k'    ][0]

# ----------------------------------------------------------
# file  = open('ds_tones.pickle', 'rb')
file  = open('ds_tones_wn.pickle', 'rb')
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

for idx, x in enumerate(ds_in):
  v[idx], state = do_filter(x, state)

plt.plot(np.arange(50), v[-50:])
plt.show()

# ----------------------------------------------------------
ds_out = ds.sinc_decimate(v, 6, OSR)

# ----------------------------------------------------------
t = np.linspace(0, 1.0/fs_to_ds, num=ds_out.shape[0], endpoint=False)

plt.plot(t, ds_out)
plt.show()

# ----------------------------------------------------------
f, Pxx_den_from_filter = signal.welch(ds_out, fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den_from_filter)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
# ----------------------------------------------------------
filter_in = ds.sinc_decimate(ds_in, 6, OSR)

# ----------------------------------------------------------
f, Pxx_den_to_filter = signal.welch(filter_in, fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den_to_filter)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
plt.semilogy(f, Pxx_den_to_filter/Pxx_den_from_filter)
# plt.semilogy(f, Pxx_den_from_filter/Pxx_den_to_filter)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


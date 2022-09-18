#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
os.environ['DISPLAY'] = ':0'
import deltasigma as ds
import pickle

# ----------------------------------------------------------
class sigma_delta():
  # ----------------------------------------------------------
  def __init__(self):
    self.x1 = 0
    self.x2 = 0

  def dsigma(self, u):
    next_x1 = self.x1 + self.x2 + u - 2*np.sign(self.x1)
    next_x2 = self.x2 + u - np.sign(self.x1)
    # The register closest to the output holds state x1 in this case.
    # The output can be written as
    data_out = np.sign(self.x1)
    self.x1 = next_x1
    self.x2 = next_x2

    return data_out


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
file  = open('ds_tones.pickle', 'rb')
# file  = open('ds_tone_1k.pickle', 'rb')
# file  = open('ds_tones_wn.pickle', 'rb')
ds_in = pickle.load(file)
file.close()

print('np.amin(ds_in)', np.amin(ds_in))
print('np.amax(ds_in)', np.amax(ds_in))

# ----------------------------------------------------------
ds_mod = sigma_delta()

def do_filter(x, state):
  next_state = np.zeros(4)

  # y  =  beta[0] * x
  # y += state[0] * k[0] * ts

  # if y > 0:
    # y = 1.0
  # else:
    # y = -1.0
    
  y_filter  =  beta[0] * x
  y_filter += state[0] * k[0] * ts
  y = ds_mod.dsigma(y_filter)

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

# # ----------------------------------------------------------
# # ds_out = ds.sinc_decimate(v, 6, OSR)
# ds_out = ds.sinc_decimate(v, 2, OSR)

#----------------------------------------------------------
sos = signal.butter(10, fb, 'lp', fs=fs_to_ds, output='sos')
ds_out = signal.sosfilt(sos, v)
ds_out = ds_out[::OSR]

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
plt.title('x')
plt.show()

# ----------------------------------------------------------
# ----------------------------------------------------------
# filter_in = ds.sinc_decimate(ds_in, 6, OSR)
filter_in = ds.sinc_decimate(ds_in, 2, OSR)

# ----------------------------------------------------------
f, Pxx_den_to_filter = signal.welch(filter_in, fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den_to_filter)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('y')
plt.show()

# # ----------------------------------------------------------
# tf_est = Pxx_den_to_filter/Pxx_den_from_filter
# tf_est /= np.amax(tf_est)

# # plt.semilogy(f[:2000], tf_est[:2000])
# plt.semilogy(f, tf_est)
# # plt.semilogy(f, Pxx_den_from_filter/Pxx_den_to_filter)
# # plt.ylim([0.5e-3, 1])
# plt.axvline(300, color='green')
# plt.axvline(3000, color='green')
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.title('y / x')
# plt.show()


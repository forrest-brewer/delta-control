#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pickle

# ----------------------------------------------------------
def psd_plot(x, Fs):
    N = x.shape[0]
    xdft = fft(x)
    xdft = xdft[0: int(N/2)]
    psdx = (1/(Fs*N)) * np.abs(xdft)**2
    psdx[1:-2] = 2*psdx[1:-2]
    freq = np.arange(0, Fs/2, Fs/N)

    plt.semilogx(freq, 10*np.log10(psdx))
    plt.grid(True)
    plt.title('PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()

    return psdx, freq

# ----------------------------------------------------------
class sigma_delta():
  # ----------------------------------------------------------
  def __init__(self):
    self.x1 = -1.0
    self.x2 = -1.0

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
# file = open('delta_sigma_cof.pickle', 'rb')
file = open('lp_filter_2k_cof.pickle', 'rb')
# file = open('cheby2_bandpass.pickle', 'rb')
delta_sigma_cof = pickle.load(file)
file.close()

# ----------------------------------------------------------
print(delta_sigma_cof)
alpha = delta_sigma_cof['alpha'][0]
beta  = delta_sigma_cof['beta' ][0]
k     = delta_sigma_cof['k'    ][0]

# ----------------------------------------------------------
# file  = open('ds_tones.pickle', 'rb')
# file  = open('ds_tone_1k.pickle', 'rb')
# file  = open('ds_tone_10k.pickle', 'rb')
# file  = open('ds_tones_wn.pickle', 'rb')
file  = open('ds_chirp.pickle', 'rb')
ds_in = pickle.load(file)
file.close()

print('np.amin(ds_in)', np.amin(ds_in))
print('np.amax(ds_in)', np.amax(ds_in))

# ----------------------------------------------------------
ds_mod = sigma_delta()

def do_filter(x, state):
  next_state = np.zeros(4)
  y_filter       =  beta[0] * x                + state[0] * k[0] * ts
  y              = ds_mod.dsigma(y_filter)
  next_state[0]  =  beta[1] * x - alpha[1] * y + state[1] * k[1] * ts + state[0]
  next_state[1]  =  beta[2] * x - alpha[2] * y + state[2] * k[2] * ts + state[1]
  next_state[2]  =  beta[3] * x - alpha[3] * y + state[3] * k[3] * ts + state[2]
  next_state[3]  =  beta[4] * x - alpha[4] * y                        + state[3]

  return y, next_state

# ----------------------------------------------------------
state = np.zeros((ds_in.shape[0] + 1, 4))
v = np.zeros(ds_in.shape[0])

for idx, x in enumerate(ds_in):
  v[idx], state[idx + 1] = do_filter(x, state[idx])

state = state[:-1,:]

# ----------------------------------------------------------
t = np.arange(ds_in.shape[0])

plt.plot(t, state[:, 0])
plt.title('state[:, 0]')
plt.show()

plt.plot(t, state[:, 1])
plt.title('state[:, 1]')
plt.show()

plt.plot(t, state[:, 2])
plt.title('state[:, 2]')
plt.show()

plt.plot(t, state[:, 3])
plt.title('state[:, 3]')
plt.show()

t = np.arange(ds_in.shape[0])
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, state[:, 0])
axs[0, 0].set_title('state[:, 0]')
axs[0, 1].plot(t, state[:, 1], 'tab:orange')
axs[0, 1].set_title('state[:, 1]')
axs[1, 0].plot(t, state[:, 2], 'tab:green')
axs[1, 0].set_title('state[:, 2]')
axs[1, 1].plot(t, state[:, 3], 'tab:red')
axs[1, 1].set_title('state[:, 3]')
plt.show()

t = np.arange(ds_in.shape[0])

plt.title('Filter States')
plt.plot(t, state[:, 3], 'tab:red', label='state[:, 3]')
plt.plot(t, state[:, 2], 'tab:green', label='state[:, 2]')
plt.plot(t, state[:, 1], 'tab:orange', label='state[:, 1]')
plt.plot(t, state[:, 0], label='state[:, 0]')
plt.legend(loc="upper left")
plt.show()

plt.title('Filter States')
# plt.plot(t, state[:, 3], 'tab:red', label='state[:, 3]')
plt.plot(t, state[:, 2], 'tab:green', label='state[:, 2]')
plt.plot(t, state[:, 1], 'tab:orange', label='state[:, 1]')
plt.plot(t, state[:, 0], label='state[:, 0]')
plt.legend(loc="upper left")
plt.show()

plt.plot(np.arange(50), v[-50:])
plt.show()

#----------------------------------------------------------
sos = signal.butter(10, fb, 'lp', fs=fs_to_ds, output='sos')
ds_out = signal.sosfilt(sos, v)
ds_out = ds_out[::OSR]

# ----------------------------------------------------------
t = np.linspace(0, 1.0/fs_to_ds, num=ds_out.shape[0], endpoint=False)
plt.plot(t, ds_out)
plt.title('Filtered and Down Sampled SD input')
plt.show()

plt.plot(t[200:350], ds_out[200:350])
plt.show()

# psd_plot(ds_out, fs)

# ----------------------------------------------------------
x, f = psd_plot(ds_in, fs_to_ds)
y, f = psd_plot(v, fs_to_ds)

# ----------------------------------------------------------
plt.loglog()
plt.plot(f[8:], y[8:] / x[8:])
plt.show()


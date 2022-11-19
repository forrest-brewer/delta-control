#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# ----------------------------------------------------------
def psd_plot(x, y, Fs):
    N = x.shape[0]

    xdft = fft(x)
    xdft = xdft[0: int(N/2)]
    psdx = (1/(Fs*N)) * np.abs(xdft)**2
    psdx[1:-2] = 2*psdx[1:-2]

    ydft = fft(y)
    ydft = ydft[0: int(N/2)]
    psdy = (1/(Fs*N)) * np.abs(ydft)**2
    psdy[1:-2] = 2*psdy[1:-2]

    freq = np.arange(0, Fs/2, Fs/N)

    plt.semilogx(freq, 10*np.log10(psdy / psdx))
    plt.grid(True)
    plt.title('PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()

    return psdx, psdy, freq

# ----------------------------------------------------------
class sigma_delta():
  def __init__(self):
    self.x1 = -1.0
    self.x2 = -1.0

  def dsigma(self, u):
    next_x1 = self.x1 + self.x2 + u - 2*np.sign(self.x1)
    next_x2 = self.x2 + u - np.sign(self.x1)
    data_out = np.sign(self.x1)
    self.x1 = next_x1
    self.x2 = next_x2

    return data_out

# ----------------------------------------------------------
def do_filter(x, state, filter, ds_mod):
  # ----------------------------------------------------------
  OSR   = filter.OSR  # oversample ratio
  fb    = filter.fb   # nyquist
  fs    = filter.fs   # sampling rate
  ts    = filter.ts   # sampling period
  alpha = filter.alpha
  beta  = filter.beta
  k     = filter.k

  # ----------------------------------------------------------
  next_state = np.zeros(filter.k.shape[0])
  y_filter       =  beta[0] * x                + state[0] * k[0] * ts
  y              = ds_mod.dsigma(y_filter)

  for i in range(1,k.shape[0]):
    next_state[i-1] = beta[i] * x - alpha[i] * y + state[i] * k[i] * ts + state[i-1]

  next_state[-1] = beta[-1] * x - alpha[-1] * y + state[-1]

  return y, next_state

# ----------------------------------------------------------
def sim_filter(filter):
  # ----------------------------------------------------------
  OSR   = filter.OSR  # oversample ratio
  fb    = filter.fb   # nyquist
  fs    = filter.fs   # sampling rate
  ts    = filter.ts   # sampling period
  alpha = filter.alpha
  beta  = filter.beta
  k     = filter.k

  # ----------------------------------------------------------
  t = np.arange(0, 0.5, 1.0/fs)
  ds_in = 0.5*signal.chirp(t,0,0.25,fb/2);

  print('np.amin(ds_in)', np.amin(ds_in))
  print('np.amax(ds_in)', np.amax(ds_in))

  # ----------------------------------------------------------
  ds_mod = sigma_delta()

  state = np.zeros((ds_in.shape[0] + 1, filter.k.shape[0]))
  v = np.zeros(ds_in.shape[0])

  for idx, x in enumerate(ds_in):
    v[idx], state[idx + 1] = do_filter(x, state[idx], filter, ds_mod)

  state = state[:-1,:]

  # ----------------------------------------------------------
  psdx, psdy, freq = psd_plot(ds_in, v, fs)

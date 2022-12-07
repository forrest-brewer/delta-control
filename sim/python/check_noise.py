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
t = np.arange(0, 0.1, 1.0/fs_to_ds)

# ----------------------------------------------------------
tx_signal = np.zeros(t.shape)

with open('noise_hex.txt', 'r') as f:
  line = f.readline()
  line = f.readline()
  line = f.readline()

  for i, x in enumerate(tx_signal):
    line = f.readline()

    if not line:
      break

    tx_signal[i] = int(line.strip(), 16)

# ----------------------------------------------------------
rx_signal = np.zeros(t.shape)

with open('rx_tb_simulink.txt', 'r') as f:
  line = f.readline()
  line = f.readline()
  line = f.readline()

  for i, x in enumerate(rx_signal):
    line = f.readline()

    if not line:
      break

    rx_signal[i] = int(line.strip(), 16)

# ----------------------------------------------------------
# util.psd_plot(tx_signal, fs_to_ds)
# util.psd_plot(rx_signal, fs_to_ds)
psdx, psdy, freq = util.tf_plot(tx_signal, rx_signal, fs_to_ds)

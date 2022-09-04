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

# ----------------------------------------------------------
file  = open('ds_tones.pickle', 'rb')
ds_in = pickle.load(file)
file.close()

# ----------------------------------------------------------
sos = signal.butter(10, fb, 'lp', fs=fs_to_ds, output='sos')
filtered = signal.sosfilt(sos, ds_in)

# ----------------------------------------------------------
t = np.linspace(0, 1.0/fs_to_ds, num=filtered.shape[0], endpoint=False)

plt.plot(t[::OSR], filtered[::OSR])
plt.show()

# ----------------------------------------------------------
f, Pxx_den_to_ds = signal.welch(filtered[::OSR], fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den_to_ds)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
ds_out = ds.sinc_decimate(ds_in, 6, OSR)

# ----------------------------------------------------------
t = np.linspace(0, 1.0/fs_to_ds, num=ds_out.shape[0], endpoint=False)

plt.plot(t, ds_out)
plt.show()

# ----------------------------------------------------------
f, Pxx_den_to_ds = signal.welch(ds_out, fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den_to_ds)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()



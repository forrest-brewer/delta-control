#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
os.environ['DISPLAY'] = ':0'
import deltasigma as ds
import pickle

# ----------------------------------------------------------
# https://python-deltasigma.readthedocs.io/en/latest/

OSR = 256
fb = 22050 # nyquist
fs = fb*2
fs_to_ds = fs*OSR # sampling rate

# ----------------------------------------------------------
# t = np.arange(0, 0.125, 1.0/fs_to_ds)

# u_to_ds  = 0.4 * np.sin(2*np.pi*1000*t) * signal.windows.hann(t.shape[0])
# u_to_ds  =  0.1 * np.sin(2*np.pi*1000*t)

# u  = SineAmp*np.sin(2*np.pi*10/Fs*np.arange(N))*ds.ds_hann(N)
# u += SineAmp*np.sin(2*np.pi*100/Fs*np.arange(N))*ds.ds_hann(N)
# u += SineAmp*np.sin(2*np.pi*500/Fs*np.arange(N))*ds.ds_hann(N)
# u += SineAmp*np.sin(2*np.pi*5000/Fs*np.arange(N))*ds.ds_hann(N)
# u0 = u_to_ds[::OSR]

# ----------------------------------------------------------
t  = np.arange(0, 2, 1.0/fs_to_ds)
u  = np.random.rand(t.shape[0])
u -= 0.5
u *= 0.4

# ----------------------------------------------------------
sos = signal.butter(10, fb, 'lp', fs=fs_to_ds, output='sos')
u_to_ds = signal.sosfilt(sos, u)


# plt.plot(np.arange(N)[::DecFact]/Fs, u0)
# plt.show()

plt.plot(t[::OSR], u_to_ds[::OSR])
plt.show()

plt.plot(t, u_to_ds)
plt.show()

# ----------------------------------------------------------
f, Pxx_den_to_ds = signal.welch(u_to_ds, fs_to_ds, nperseg=1024*8)
# plt.semilogy(f/OSR, Pxx_den_to_ds)
plt.semilogy(f, Pxx_den_to_ds)
# plt.ylim([50, fb])
# plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
f, Pxx_den_to_ds = signal.welch(u_to_ds[::OSR], fs, nperseg=1024*8)
# plt.semilogy(f/OSR, Pxx_den_to_ds)
plt.semilogy(f, Pxx_den_to_ds)
# plt.xlim([50, 5000])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
H = ds.synthesizeNTF(5, OSR, 1)
v = ds.simulateDSM(u_to_ds, H)[0]

# ----------------------------------------------------------
file = open('ds_tones.pickle', 'wb')
pickle.dump(v, file)
file.close()
print('DeltaSigma tones done!')

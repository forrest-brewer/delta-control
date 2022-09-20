#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pickle

# ----------------------------------------------------------

OSR = 256
fb = 22050 # nyquist
fs = fb*2
fs_to_ds = fs*OSR # sampling rate

# ----------------------------------------------------------
# function [psdx, freq] = psd(x,Fs)
def psd_plot(x, Fs):

    # x = input signal
    # Fs = sampling frequency

    # N = length(x)
    N = x.shape[0]
    # xdft = fft(x)
    xdft = fft(x)
    # xdft = xdft(1:N/2+1)
    xdft = xdft[0: int(N/2)]
    # psdx = (1/(Fs*N)) * abs(xdft).^2
    psdx = (1/(Fs*N)) * np.abs(xdft)**2
    # psdx(2:end-1) = 2*psdx(2:end-1)
    psdx[1:-2] = 2*psdx[1:-2]
    # freq = 0:Fs/length(x):Fs/2
    freq = np.arange(0, Fs/2, Fs/N)

    # semilogx(freq, 10*log10(psdx))
    plt.semilogx(freq, 10*np.log10(psdx))
    # ylim([-120 20]) # you may want to adjust the y limit here
    plt.grid(True)
    plt.title('PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()

    return psdx, freq
# end

# # ----------------------------------------------------------
# def ds_mod(data_in):
  # x1 = 0
  # x2 = 0
  # data_out = np.zeros(data_in.shape[0])
  # next_x1 = 0
  # next_x2 = 0

  # for n, u in enumerate(data_in):
    # next_x1 = x1 + x2 + u - 2*np.sign(x1)
    # next_x2 = x2 + u - np.sign(x1)
    # # The register closest to the output holds state x1 in this case.
    # # The output can be written as
    # data_out[n] = np.sign(x1)
    # x1 = next_x1
    # x2 = next_x2

  # return data_out

# ----------------------------------------------------------
# t = np.arange(0, 0.25, 1.0/fs_to_ds)
t = np.arange(0, 2.5, 1.0/fs_to_ds)

u = 1.0 * np.sin(2*np.pi* 5000 * t) * signal.windows.hann(t.shape[0])

# u  = 1.0 * np.sin(2*np.pi*  100 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi*  500 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi* 1000 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi* 2500 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi* 5000 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi*10000 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi*15000 * t) * signal.windows.hann(t.shape[0])
# u += 1.0 * np.sin(2*np.pi*20000 * t) * signal.windows.hann(t.shape[0])

u -= np.amin(u)
u /= np.amax(u)
u *= 2
u -= 1
u *= 0.5

print('np.amin(u)', np.amin(u))
print('np.amax(u)', np.amax(u))

# ----------------------------------------------------------
f, Pxx_den_from_u = signal.welch(u[::OSR], fs, nperseg=1024*8)
plt.plot(f, Pxx_den_from_u)
plt.loglog()
plt.title('u[::OSR]')
plt.show()

# # ----------------------------------------------------------
# v = ds_mod(u)

#----------------------------------------------------------
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

#----------------------------------------------------------
ds_mod = sigma_delta()
state = np.zeros(4)
v = np.zeros(u.shape[0])

for idx, x in enumerate(u):
  v[idx] = ds_mod.dsigma(x)

# # ----------------------------------------------------------
# Pxx_den_from_filter, f = plt.psd(v, 1024*8, fs_to_ds)
# plt.plot(f, Pxx_den_from_filter)
# plt.loglog()
# plt.title('5kHz sine wave @ DS modulator output')
# plt.show()

# ----------------------------------------------------------
psd_plot(v, fs_to_ds)

# # ----------------------------------------------------------
# file = open('ds_tones.pickle', 'wb')
# pickle.dump(v, file)
# file.close()
# print('DeltaSigma tones done!')


#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle

# ----------------------------------------------------------

OSR = 256
fb = 22050 # nyquist
fs = fb*2
fs_to_ds = fs*OSR # sampling rate

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
t = np.arange(0, 0.25, 1.0/fs_to_ds)

u  = 1.0 * np.sin(2*np.pi*  100 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi*  500 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi* 1000 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi* 2500 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi* 5000 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi*10000 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi*15000 * t) * signal.windows.hann(t.shape[0])
u += 1.0 * np.sin(2*np.pi*20000 * t) * signal.windows.hann(t.shape[0])

u -= np.amin(u)
u /= np.amax(u)
u *= 2
u -= 1
# u *= 0.1

print('np.amin(u)', np.amin(u))
print('np.amax(u)', np.amax(u))

# ----------------------------------------------------------
t = np.linspace(0, 1.0/fs_to_ds, num=u.shape[0], endpoint=False)

plt.plot(t, u)
plt.show()

# ----------------------------------------------------------
f, Pxx_den_from_u = signal.welch(u[::OSR], fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den_from_u)
# plt.ylim([0.5e-3, 1])
plt.axvline(1000, color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
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
plt.title('ds_mod after LP filter and down sample')
plt.show()

# # ----------------------------------------------------------
# plt.semilogy(f, Pxx_den_from_filter / Pxx_den_from_u)
# # plt.ylim([0.5e-3, 1])
# plt.axvline(1000, color='green')
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.title('y / x')
# plt.show()

# ----------------------------------------------------------
file = open('ds_tones.pickle', 'wb')
pickle.dump(v, file)
file.close()
print('DeltaSigma tones done!')


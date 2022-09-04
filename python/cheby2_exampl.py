# ----------------------------------------------------------
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ----------------------------------------------------------
b, a = signal.cheby2(4, 40, 100, 'low', analog=True)
w, h = signal.freqs(b, a)

plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Chebyshev Type II frequency response (rs=40)')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axhline(-40, color='green') # rs
plt.show()

# ----------------------------------------------------------
sample_time = 100
fs = 1000
t = np.linspace(0, sample_time, fs*sample_time, False)
# sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
sig = np.random.rand(t.shape[0])
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('10 Hz and 20 Hz sinusoids')
ax1.axis([0, 1, -2, 2])

# ----------------------------------------------------------
sos = signal.cheby2(12, 20, 17, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After 17 Hz high-pass filter')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.show()

# ----------------------------------------------------------
f_in, Pxx_den_in = signal.welch(sig, fs, nperseg=1024*8)
plt.semilogy(f_in, Pxx_den_in)
# plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('sig PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
f, Pxx_den = signal.welch(filtered, fs, nperseg=1024*8)
plt.semilogy(f, Pxx_den)
# plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('filtered PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
f_in, Pxx_den_in = signal.welch(sig, fs, nperseg=1024*8)
# Pxx_den = Pxx_den / Pxx_den_in
plt.semilogy(f, Pxx_den / Pxx_den_in)
# plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('y/x PSD [V**2/Hz]')
plt.show()

# ----------------------------------------------------------
f_in, Pxx_den_in = signal.welch(sig, fs, nperseg=1024*8)
Pxx_den = Pxx_den / Pxx_den_in
plt.semilogy(f[0:200], Pxx_den[0:200])
# plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('y/x PSD [V**2/Hz]')
plt.show()



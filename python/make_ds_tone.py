#
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['DISPLAY'] = ':0'
import deltasigma as ds
import pickle

# ----------------------------------------------------------
# https://python-deltasigma.readthedocs.io/en/latest/
SineAmp  = 0.1
SineFreq = 1000

SincOrder = 8
DecFact   = 256

T = 0.125 # Input signal duration in seconds.
fb = 22050 # nyquist
# FsOut = 8192 # set to ensure compatibility.
FsOut = 22050*2
Fs = FsOut*DecFact #Fs

N = int(np.round(T*Fs))

u  = SineAmp*np.sin(2*np.pi*10/Fs*np.arange(N))*ds.ds_hann(N)
u += SineAmp*np.sin(2*np.pi*100/Fs*np.arange(N))*ds.ds_hann(N)
u += SineAmp*np.sin(2*np.pi*500/Fs*np.arange(N))*ds.ds_hann(N)
# u += SineAmp*np.sin(2*np.pi*5000/Fs*np.arange(N))*ds.ds_hann(N)
u0 = u[::DecFact]

plt.plot(np.arange(N)[::DecFact]/Fs, u0)
plt.show()

# ----------------------------------------------------------
N = max(u0.shape)
U = np.fft.fft(u0)/(N/4)
f = np.linspace(0, FsOut, N + 1)
f = f[:int(N/2 + 1)]
plt.semilogx(f, ds.dbv(U[:int(N/2) + 1]))
plt.xlabel('f [Hz]')
plt.ylabel('U(f) [dB]')
plt.show()

# ----------------------------------------------------------
ABCD = np.array([[1., 0., 1., -1.], [1., 1., 1., -2.], [0., 1., 0., 0.]])
v, junk1, junk2, y = ds.simulateDSM(u, ABCD)
del junk1, junk2
q = v - y # quantization error

# ----------------------------------------------------------
file = open('ds_tones.pickle', 'wb')
pickle.dump(v, file)
file.close()

# ----------------------------------------------------------
N = max(v.shape)
nPlot = 2000*64
if N > nPlot:
    n = np.arange(int(np.floor(N/2 - nPlot/2)), int(np.floor(N/2 + nPlot/2)))
else:
    n = np.arange(N)
n = n.astype(np.int32)
# hold(True)
t = np.arange(max(n.shape))
plt.step(t, u[n], 'r')
plt.bar(t, v[n], color='b', linewidth=0)
plt.ylabel('$u(t), v(t)$')
plt.xlabel('Sample #')
plt.axis([0, max(n)-min(n), -1.1, 1.1])
# figureMagic(size=(20, 4), name='Modulator Input & Output')
plt.show()

# ----------------------------------------------------------
N = max(v.shape)
Nfft = min(N, 16*8192)
n = np.arange((N - Nfft)/2, (N + Nfft)/2).astype(np.int32)
V = np.fft.fft(v[n] * ds.ds_hann(Nfft)) / (Nfft / 4)
inBin = np.ceil(Nfft/1000)
# hold(True)
plt.ylabel('V(f) [dB]')
plt.xlabel('Frequency [Hz]')
plt.semilogx(np.arange(max(V.shape))/max(V.shape)*Fs, ds.dbv(V))
f, Vp = ds.logsmooth(V, inBin)
plt.semilogx(f*Fs, Vp, 'm', linewidth=2.5)
plt.xlim([f[0]*Fs, Fs/2])
msg = 'NBW = %.1f Hz ' % (Fs*1.5/Nfft)
plt.text(Fs/2, -90, msg, horizontalalignment='right', verticalalignment='center')
# figureMagic(size=plotsize, name='Spectrum')
plt.show()

# # ----------------------------------------------------------
w = ds.sinc_decimate(v, SincOrder, DecFact)
filtered_q = ds.sinc_decimate(q, SincOrder, DecFact)

# N = max(w.shape)
# t = np.arange(N)/FsOut
# plt.subplot(211)
# plt.plot(t, w)
# plt.ylabel('$w$')
# # figureMagic(size=(20, 4))
# plt.show()

# plt.subplot(212)
# plt.plot(t, u0 - w, 'g')
# plt.ylabel('$u-w$')
# plt.xlabel('t [s]')
# # figureMagic(size=(20, 4))
# plt.suptitle('Output and conversion error');
# plt.show()

# ----------------------------------------------------------
N = max(filtered_q.shape)
Nfft = min(N, 16*8192)
n = np.arange((N - Nfft)/2, (N + Nfft)/2).astype(np.int32)
E = np.fft.fft(filtered_q[n] * ds.ds_hann(Nfft)) / (Nfft / 4)
W = np.fft.fft(w[n] * ds.ds_hann(Nfft)) / (Nfft / 4)
U0 = np.fft.fft(u0[n] * ds.ds_hann(Nfft)) / (Nfft / 4)
inBin = np.round(SineFreq*Nfft)/FsOut
# hold(True)
plt.ylabel('dB')
plt.semilogx(np.arange(Nfft)/Nfft*FsOut, ds.dbv(U0), label='Input signal')
plt.semilogx(np.arange(Nfft)/Nfft*FsOut, ds.dbv(E), label='Filtered quant. error')
plt.semilogx(np.arange(Nfft)/Nfft*FsOut, ds.dbv(W), label='Output signal')
f, U0p = ds.logsmooth(U0, inBin)
plt.semilogx(f*FsOut, U0p, '#1E90FF', linewidth=2.5)
f, Ep = ds.logsmooth(E, inBin)
plt.semilogx(f*FsOut, Ep, '#8B0000', linewidth=2.5)
f, Wp = ds.logsmooth(W, inBin)
plt.semilogx(f*FsOut, Wp, '#556B2F', linewidth=2.5)
plt.xlim([10, FsOut/2])
msg = 'NBW = %.1f Hz ' % (Fs*1.5/Nfft)
plt.text(FsOut/2, -6, msg, horizontalalignment='right', verticalalignment='top')
# figureMagic(size=plotsize, name='Spectrum')
plt.legend(loc=3);
plt.show()

# ----------------------------------------------------------
# nPlot = 100*64
# n = np.arange(int(np.floor(N/2 - nPlot/2)), int(np.floor(N/2 + nPlot/2)))
n = np.arange(N)
n = n.astype(np.int32)
t = np.arange(max(n.shape))
plt.plot(t, w[n], 'r')
plt.ylabel('$w(t)$')
plt.xlabel('Sample #')
plt.axis([0, max(n)-min(n), -1.1, 1.1])
plt.show()

# ----------------------------------------------------------
N = max(w.shape)
U = np.fft.fft(w)/(N/4)
f = np.linspace(0, FsOut, N + 1)
f = f[:int(N/2 + 1)]
plt.semilogx(f, ds.dbv(U[:int(N/2) + 1]))
plt.xlabel('f [Hz]')
plt.ylabel('U(f) [dB]')
plt.show()


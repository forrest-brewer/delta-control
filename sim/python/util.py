#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

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
def tf_plot(x, y, Fs, xlim=None):
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

    plt.semilogx(freq, 10*np.log10(psdy/psdx))
    plt.grid(True)
    plt.title('PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')

    if xlim is not None:
      plt.xlim(xlim)

    plt.show()

    return psdx, psdy, freq

# ----------------------------------------------------------
def int2hex(number, bits):
  """ Return the 2'complement hexadecimal representation of a number """
  if number < 0:
    x = (1 << bits) + number
  else:
    x = number

  return f'{int(x):x}'

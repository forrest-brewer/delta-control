#
import numpy as np
import matplotlib.pyplot as plt
import pickle
from fxpmath import Fxp

# ----------------------------------------------------------
file = open('cheby2_bandpass.pickle', 'rb')
delta_sigma_cof = pickle.load(file)
file.close()

# ----------------------------------------------------------
OSR = 256     # oversample ratio
fb = 22050    # nyquist
fs = OSR*2*fb # sampling frequency
ts = 1/fs     # sampling period

# ----------------------------------------------------------
# print(delta_sigma_cof)
alpha = delta_sigma_cof['alpha']
beta  = delta_sigma_cof['beta' ]
k     = delta_sigma_cof['k'    ]

# ----------------------------------------------------------
# https://github.com/Schweitzer-Engineering-Laboratories/fixedpoint

alpha_fp = Fxp(alpha)
beta_fp  = Fxp(beta)
k_fp     = Fxp(k*ts)

a = alpha_fp.bin(frac_dot=True)[0]
b = beta_fp.bin(frac_dot=True)[0]
k = k_fp.bin(frac_dot=True)[0]

for idx, j in enumerate(k):
  print('-'*20, 'index', idx)
  print('a[idx]', a[idx])
  print('b[idx]', b[idx])
  print('k[idx]', k[idx])
  
print('-'*20, 'index', a.shape[0] - 1)
print('a[idx]', a[-1])
print('b[idx]', b[-1])

# ----------------------------------------------------------
k_fp.info(verbose=3)
print(k_fp.hex())



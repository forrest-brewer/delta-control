import numpy as np
from scipy import signal
from scipy import linalg
import control
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import Math, display, Latex

import sys

if '../sdfpy' not in sys.path:
  sys.path.insert(0,'../sdfpy')

import sd_sim
import sdfpy as sdf

OSR = 256      # oversample ratio
fb = 22050     # nyquist
fs = OSR*2*fb  # sampling frequency
ts = 1/fs      # sampling period

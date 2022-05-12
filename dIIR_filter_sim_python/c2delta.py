#
#
import numpy as np
from scipy import signal
from scipy import linalg


# function [Ad,Bd,Cd,Dd] = c2delta(A,B,C,D,ts);
def c2delta(A,B,C,D,ts):

  # Ad = (expm(A*ts)-eye(size(A)))./ts;
  Ad = (linalg.expm(A*ts) - np.eye(A.shape[0])) / ts
  
  # Bd = A\((expm(A*ts)-eye(size(A)))*B)./ts;
  Bd = np.matmul((linalg.expm(A*ts) - np.eye(A.shape[0]) ), B) / ts
  Bd = np.matmul(np.linalg.inv(A), Bd)
  
  Cd = C;
  Dd = D;
  
  return Ad,Bd,Cd,Dd

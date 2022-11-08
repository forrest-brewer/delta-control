#
#
import numpy as np
from scipy import signal
from scipy import linalg
import control

# ----------------------------------------------------------
def c2delta(A,B,C,D,ts):
  Ad = (linalg.expm(A*ts) - np.eye(A.shape[0])) / ts
  Bd = np.matmul((linalg.expm(A*ts) - np.eye(A.shape[0]) ), B) / ts
  Bd = np.matmul(np.linalg.inv(A), Bd)
  Cd = C;
  Dd = D;
  return Ad,Bd,Cd,Dd

# ----------------------------------------------------------
def obsv_cst(A,B,C,D):
  e     = np.zeros((A.shape[1], 1))
  e[-1] = 1

  O = control.obsv(A, C)
  [U, S, Vh] = linalg.svd(O)
  V = Vh.T
  S = np.diag(S)

  S_inv = linalg.solve(S, np.eye(S.shape[0]))
  T_inv = V @ S_inv @ U.conj().T

  T1 = T_inv @ e;
  n  = A.shape[1]
  q  = T1.shape[1]
  T0 = np.zeros((n,n))

  for i in range(1, n+1):
    column = np.power(A, n-i) @ T1
    T0[:, i-1] = column[:,0]

  AT = linalg.solve(T0, A @ T0)
  BT = linalg.solve(T0, B)
  CT = C @ T0;
  DT = D;

  return AT,BT,CT,DT,T0

# ----------------------------------------------------------
def delta_bode(A,B,C,D,f,ts):
  q     = C.shape[0]
  p     = B.shape[1]
  fs    = 1/ts
  mag   = np.zeros((q,p,f.shape[0]))
  phz   = np.zeros((q,p,f.shape[0]))
  delta = (np.exp(1j*2*np.pi*(f/fs))-1)/ts

  for i in range(f.shape[0]):
      A_d = delta[i] * np.eye(A.shape[0]) - A
      h   = C @ linalg.solve(A_d, B) + D
      mag[:,:,i] = np.abs(h)
      phz[:,:,i] = 180*np.arctan2(np.imag(h),np.real(h))/np.pi
  return mag, phz

# ----------------------------------------------------------
def dIIR_scaling(A,B,T0,f,ts):
  T0_inv = linalg.solve(T0, np.eye(T0.shape[0]))
  [f_int, phz] = delta_bode(A,B,T0_inv,0,f,ts)

  f_norm = np.zeros(A.shape[0])

  for i in range(f_norm.shape[0]):
    f_norm[i] = linalg.norm(f_int[i], np.inf, axis=1)

  Ts = np.zeros(A.shape)
  k = np.zeros(f_norm.shape[0])
  k_inv = np.zeros(f_norm.shape[0])

  for i in range(f_norm.shape[0]):
    if i == 0:
      k[i] = 1/f_norm[i]
    else:
      k[i] = 1/(np.prod(k[:i])*f_norm[i])

    k_inv[i] = 2**np.floor(np.log2(ts/k[i]))/ts
    Ts[i,i] = np.prod(k_inv[0:i+1])

  return Ts, k_inv
  
# ----------------------------------------------------------
def ss_concat_outputs(sys_0, sys_1):
  A = linalg.block_diag(sys_0.A, sys_1.A)
  B = np.vstack((sys_0.B       , sys_1.B))
  C = linalg.block_diag(sys_0.C, sys_1.C)
  D = np.vstack((sys_0.D       , sys_1.D))
  return control.ss(A,B,C,D)

# ----------------------------------------------------------
def SD_dIIR_sensitivity(A,B,C,D,T0,Ts,f,ts):
  # %Sensitivity due to feedthrough coefficient
  B_beta = np.zeros((1, A.shape[1])).T
  B_beta[0][0] = 1
  B_beta = A @ T0 @ B_beta
  S_beta0 = control.ss(A,B_beta,C,1)

  # %Sensitivity due to numerator coefficients
  S_beta = control.ss(A.T,C.T,Ts.T @ T0.T,0)

  # %Sensitivity due to denominator coefficients
  H = control.ss(A,B,C,D)
  S_alpha = control.series(H, S_beta)
  S_bsys = ss_concat_outputs(S_beta0, S_beta)
  null_sys = control.ss(0, 0, 0, 0)
  S_asys = ss_concat_outputs(null_sys, S_alpha)

  [S_bmag,S_bphz] = delta_bode(S_bsys.A,S_bsys.B,S_bsys.C,S_bsys.D,f,ts)
  [S_amag,S_aphz] = delta_bode(S_asys.A,S_asys.B,S_asys.C,S_asys.D,f,ts)
  S_mag = np.squeeze(S_bmag + S_amag)
  S_phz = np.squeeze(S_bphz + S_aphz)

  return S_mag, S_phz


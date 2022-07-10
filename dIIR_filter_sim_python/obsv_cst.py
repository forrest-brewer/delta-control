#
#
import numpy as np
from scipy import signal
from scipy import linalg
import control
from numpy import linalg as LA

# function [AT,BT,CT,DT,T0] = obsv_cst(A,B,C,D)
def obsv_cst(A,B,C,D):

  # e = zeros(size(A,2),1);
  e = np.zeros((A.shape[1], 1))

  # e(end) = 1;
  e[-1] = 1

  # O = obsv(A,C);
  O = control.obsv(A, C)

  # [U, S, V] = svd(O);
  [U, S, Vh] = linalg.svd(O)
  V = Vh.T
  S = np.diag(S)

  # S_inv = S\eye(size(S));
  # S_inv = np.matmul(np.linalg.inv(S), np.eye(S.shape[0]))
  # S_inv = np.linalg.inv(S) @ np.eye(S.shape[0])
  S_inv = linalg.solve(S, np.eye(S.shape[0]))

  # T_inv = V*S_inv*U';
  # T_inv = np.matmul(np.matmul(V, S_inv), U)
  T_inv = V @ S_inv @ U.conj().T

  # T1 = T_inv*e;
  # T1 =  np.matmul(T_inv, e)
  T1 = T_inv @ e;

  # n = size(A,2);
  # q = size(T1,2);
  n = A.shape[1]
  q = T1.shape[1]

  # T0 = zeros(n,q);
  # T0 = np.zeros((n,q))
  T0 = np.zeros((n,n))

  # for i = 1:n
      # T0(:,q*(i-1)+1:q*i) = A^(n-i)*T1;
  # end

  for i in range(1, n+1):
    column = np.power(A, n-i) @ T1
    T0[:, i-1] = column[:,0]

  # AT = T0\(A*T0);
  # AT = np.linalg.inv(T0) @ (A @ T0)
  AT = linalg.solve(T0, A @ T0)

  # # AT(:,2:size(AT,2)) = [eye(n-1);zeros(1,n-1)];
  # # AT[:,1:AT.shape[1]] = np.block([np.eye(n-1), np.zeros(1,n-1)])
  # AT[:n-1,1:n] = np.eye(n-1)
  # AT[n-1:n,1:n] = np.zeros((1,n-1))

  # BT = T0\B;
  # BT = np.linalg.inv(T0) @ B
  BT = linalg.solve(T0, B)

  # CT = C*T0;
  CT = C @ T0;

  DT = D;

  return AT,BT,CT,DT,T0

#
#
import numpy as np
from scipy import signal
from scipy import linalg
import control
from numpy import linalg as LA
from scipy.linalg import expm_cond

# function [AT,BT,CT,DT,T0] = obsv_cst(A,B,C,D)
def obsv_cst(A,B,C,D):

  # e = zeros(size(A,2),1);
  e = np.zeros((A.shape[1], 1))
  # print(e)
  # print('e.shape', e.shape)

  # e(end) = 1;
  e[-1] = 1
  
  # print(e)
  # print('e.shape', e.shape)

  print('LA.cond(A)  ', LA.cond(A))
  print('expm_cond(A)', expm_cond(A))
  
  # O = obsv(A,C);
  O = control.obsv(A, C)
  # print(O)
  # print('O.shape', O.shape)
  
  print('LA.cond(O)  ', LA.cond(O))
  print('expm_cond(O)', expm_cond(O))

  # [U, S, V] = svd(O);
  [U, S, Vh] = linalg.svd(O)
  V = Vh.T
  S = np.diag(S)
  
  # print('U.shape', U.shape)
  # print(U)
  # print('S.shape', S.shape)
  # print(S)
  # print('V.shape', V.shape)
  # print(V)

  # S_inv = S\eye(size(S));
  # S_inv = np.matmul(np.linalg.inv(S), np.eye(S.shape[0]))
  # S_inv = np.linalg.inv(S) @ np.eye(S.shape[0])
  S_inv = linalg.solve(S, np.eye(S.shape[0])) 
  
  # print('S_inv.shape', S_inv.shape)
  # print(S_inv)

  # T_inv = V*S_inv*U';
  # T_inv = np.matmul(np.matmul(V, S_inv), U)
  T_inv = V @ S_inv @ U.conj().T
  
  # print('T_inv.shape', T_inv.shape)
  # print(T_inv)

  # T1 = T_inv*e;
  # T1 =  np.matmul(T_inv, e)
  T1 = T_inv @ e;

  # print('T1.shape', T1.shape)
  # print(T1)

  # n = size(A,2);
  # q = size(T1,2);
  n = A.shape[1]
  q = T1.shape[1]
  
  # print('n, q', n, q)

  # T0 = zeros(n,q);
  # T0 = np.zeros((n,q))
  T0 = np.zeros((n,n))

  # for i = 1:n
      # T0(:,q*(i-1)+1:q*i) = A^(n-i)*T1;
  # end
  
# **** 1 | 1, 1, 3
# T0(:,1:1) = A^(3)*T1;
# **** 2 | 2, 2, 2
# T0(:,2:2) = A^(2)*T1;
# **** 3 | 3, 3, 1
# T0(:,3:3) = A^(1)*T1;
# **** 4 | 4, 4, 0
# T0(:,4:4) = A^(0)*T1;

  for i in range(1, n+1):
    column = np.power(A, n-i) @ T1
    T0[:, i-1] = column[:,0]
    print('####', i, q*(i-1)+1, q*i, n-i)
    print(T0)
    # print(np.power(A, n-i) @ T1)

  print('T0.shape', T0.shape)
  print('np.linalg.cond(T0)', np.linalg.cond(T0))
  # print(T0)

  # AT = T0\(A*T0);
  # AT = np.linalg.inv(T0) @ (A @ T0)
  AT = linalg.solve(T0, A @ T0) 

  # print(AT)
  # print('AT.shape', AT.shape)

  # AT(:,2:size(AT,2)) = [eye(n-1);zeros(1,n-1)];
  # AT[:,1:AT.shape[1]] = np.block([np.eye(n-1), np.zeros(1,n-1)])
  AT[:n-1,1:n] = np.eye(n-1)
  AT[n-1:n,1:n] = np.zeros((1,n-1))
  
  # print(AT)
  # print('AT[1:,0:n-1].shape', AT[1:,0:n-1].shape)
  
  # BT = T0\B;
  # BT = np.linalg.inv(T0) @ B
  BT = linalg.solve(T0, B) 

  # CT = C*T0;
  CT = C @ T0;

  DT = D;

  # end


  return AT,BT,CT,DT,T0

#
import numpy as np
import scipy.io as sio
import pickle

# ----------------------------------------------------------
file_name = './cheby2_bandpass/all.mat'

mat_data = sio.loadmat(file_name)
# print(mat_data)

for key, value in mat_data.items():
  # print(key, value)
  print(key)

# ----------------------------------------------------------
delta_sigma_cof = {}
delta_sigma_cof['alpha'] = mat_data['alpha']
delta_sigma_cof['beta' ] = mat_data['beta' ]
delta_sigma_cof['k'    ] = mat_data['k'    ]
print(delta_sigma_cof)

# ----------------------------------------------------------
# open a file, where you want to store the data
file = open('delta_sigma_cof.pickle', 'wb')

# dump information to that file
pickle.dump(delta_sigma_cof, file)

# close the file
file.close()

# ----------------------------------------------------------
# open a file, where you stored the pickled data
file = open('delta_sigma_cof.pickle', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print(data)


from random import randint
import numpy as np
from scipy.sparse import csr_matrix
import csv

def sig(x):
    return 1.0/(1.0+np.exp(-x))

traindata = np.load('train_x_sparse.npz')
traindata = csr_matrix((traindata['data'], traindata['indices'], traindata['indptr']),
                      shape=traindata['shape'])

# outputs
with open('train_y.csv') as f:
    trainy = f.readlines()
totaly = []
for i, s in enumerate(trainy):
    index = int(trainy[i].replace('\n', ''))
    tempy = np.zeros(10)
    tempy[index] = 1.0
    totaly.append(tempy)
trainy = totaly

idx = 2000
inputx = (1/256)*traindata.getrow(idx).toarray() + 0.01
y = trainy[idx]

w1 = np.load('w1-500-n.npy')
w2 = np.load('w2-500-n.npy')

h_in = inputx.dot(w1)
h_out = sig(h_in)
o_in = h_out.dot(w2)
o = sig(o_in)

print(o)
print(y)

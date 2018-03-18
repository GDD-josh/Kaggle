from random import randint
import numpy as np
from scipy.sparse import csr_matrix
import csv
import math

# sigmoid function
def sig(x):
    return 1.0/(1.0+np.exp(-x))

# derivative of sigmoid
def derivative(x):
    return x*(1-x)

#loading sparse training data
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

sizex = traindata.shape[1]
# scaling factor in backpropagation
eps = 0.5
# number of neurons in the hidden layer
hiddenneurons = 500
w1 = np.random.uniform(low=-(1/math.sqrt(sizex)), high=(1/math.sqrt(sizex)), size=(sizex, hiddenneurons))
w2 = np.random.uniform(low=-(1/math.sqrt(hiddenneurons)), high=(1/math.sqrt(hiddenneurons)), size=(hiddenneurons, 10))

print('training...')

totalerr = 0
for epoch in range (0, 50000):

    # Take random training sample
    randomidx = randint(0, traindata.shape[0]-1)
    randomx = traindata.getrow(randomidx).toarray() + 0.01
    randomy = trainy[randomidx]

    #forward
    h_in = randomx.dot(w1)
    h_out = sig(h_in)
    o_in = h_out.dot(w2)
    o = sig(o_in)

    #back
    o_err = randomy - o

    o_delta = o_err*derivative(o)
    h_err = o_delta.dot(w2.T)
    h_delta = h_err*derivative(h_out)
    w1 += eps*randomx.T.dot(h_delta)
    w2 += eps*h_out.T.dot(o_delta)

    err = 0
    for e in o_err[0]:
        err += (e**2)/2

    totalerr += err
    if epoch % 1000 == 0:
        avgerr = totalerr/1000
        totalerr = 0
        print('epoch: ' + str(epoch) + '/50000 ... error: ' + str(avgerr))
        np.save('w1-500-n.npy', w1)
        np.save('w2-500-n.npy', w2)

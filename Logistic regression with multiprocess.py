from sklearn import datasets
import numpy as np
from scipy import sparse
import cPickle
import multiprocessing as mp
import numpy as np
import time
# Question 2
fin = open("data_files.pl", "rb");
data = cPickle.load(fin);
xtrain = data[0]
ytrain = data[1]
xtest = data[2]
ytest = data[3]

# prediction
def accuracy(xtest,ytest,wtrain):
    xtest = sparse.csr_matrix(xtest)
    wtrain = sparse.csr_matrix(wtrain)
    s = np.dot(xtest,wtrain)
    s = sparse.csr_matrix.todense(s)
    count = 0
    f = np.exp(s)
    f = 1 + f
    f = 1 / f
    ym = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if(f[i]>0.5):
            ym[i] = 1
        else:
            ym[i] = -1
    cam = ym.shape[0]
    count = 0
    for i in range(ytest.shape[0]):
        if (ytest[i] == ym[i]):
            count = count + 1

    print((float(count)/(ytest.shape[0]))*100)




#gradient for logistic regression
def deltaa(w,y_array,x_array,n):
    yd = np.diag(y_array)
    yd = sparse.csr_matrix(yd)
    x_array = sparse.csr_matrix(x_array)
    q = np.dot(yd,x_array)
    w = sparse.csr_matrix(w)
    a = np.dot(q,w)
    c = sparse.csr_matrix.todense(a)
    c = np.exp(c)
    c = c+1
    c = 1/c
    qt = q.transpose()
    c = sparse.csr_matrix(c)
    a = [[]]*5
    for i in range(4):
        a[i] = qt[(i*25):((i+1)*25),:]
    a[4] = qt[100:122,:]
    pool = mp.Pool(processes=4)
    result = [pool.apply(parllelmultiplication, args=(a[x], c)) for x in range(5)]
    for i in range(5):
        result[i] = sparse.csc_matrix.todense(result[i])
    a = np.vstack((result[0],result[1],result[2],result[3],result[4]))
    w = sparse.csr_matrix.todense(w)
    df = ((a / n) + w)
    return df

def parllelmultiplication(a,b):
    return np.dot(a,b)
#wtrain or gradient descent for logistic regression
def grad1(n,e,w,iter,y_array,x_array):
    wa = w
    for i in range(iter):
        wa = wa - n * deltaa(wa, y_array, x_array, xtest.shape[0])
    return wa

start_time = time.time()
print("please wait till the training is done")
wtrain = grad1(0.0001,0.01,np.random.rand(xtest.shape[1],1),100,ytrain,xtrain)
print("Time %lf secs.\n", time.time() - start_time)
accuracy(xtest,ytest,wtrain)

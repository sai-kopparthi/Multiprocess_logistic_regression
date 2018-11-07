import cPickle
import multiprocessing as mp
import numpy as np
import time

fin = open("data_files.pl", "rb")
data = cPickle.load(fin)
Xtrain = data[0]
ytrain = data[1]
Xtest = data[2]
ytest = data[3]
xtest = [[],[],[],[]]
ytestr = [[],[],[],[]];
xtest[0] = Xtest[0:250,:]
xtest[1] = Xtest[250:500:]
xtest[2] = Xtest[500:750,:]
xtest[3] = Xtest[750:1000,:]
ytestr[0] = ytest[0:250]
ytestr[1] = ytest[250:500]
ytestr[2] = ytest[500:750]
ytestr[3] = ytest[750:1000]

def go_nn(Xtrain, ytrain, Xtest, ytest,q):
    correct =0
    for i in range(Xtest.shape[0]): ## For all testing instances
        nowXtest = Xtest[i,:]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0,:]-nowXtest)
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest-Xtrain[j,:])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor

        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:
            correct += 1
    acc = correct/float(Xtest.shape[0])
    q.put(acc)

output = mp.Queue()

start_time = time.time()
procs = [mp.Process(target=go_nn, args=(Xtrain, ytrain , xtest[i] , ytestr[i], output)) for i in range(4) ]

# Run processes
for p in procs:
    p.start()

# Exit the completed processes
for p in procs:
    p.join()

results = [output.get() for i in range(4)]
results = sum(results)
results = results/4 
print(results)
print ("Time %lf secs.\n",time.time()-start_time)


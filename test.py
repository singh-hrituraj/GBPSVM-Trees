import numpy as np 
from ensembles.RF import RF
from ensembles.GBT import GBT
from estimators.CART import CART
import scipy.io as io

RF = GBT(n_estimators =5, nvartosample=4, mode='hyperplane_psvm')

dataX = np.array(io.loadmat('iris.mat')['dataX'])
dataY = np.array(io.loadmat('iris.mat')['dataY']).squeeze()

# dataX = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3]])
# dataY = np.array([-1,-1,-1,1,1,1])
RF.train(dataX, dataY)


predict  = RF.predict(dataX)

accuracy = np.sum(predict==dataY)/float(len(dataY))

print(accuracy)




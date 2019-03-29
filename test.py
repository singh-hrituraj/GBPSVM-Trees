import numpy as np 
from ensembles.RF import RF
from estimators.CART import CART
import scipy.io as io


tree  = CART(nvartosample=4, mode='hyperplane_psvm')

dataX = np.array(io.loadmat('iris.mat')['dataX'])
dataY = np.array(io.loadmat('iris.mat')['dataY']).squeeze()

# dataX = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3]])
# dataY = np.array([-1,-1,-1,1,1,1])
tree.train(dataX, dataY)


predict  = tree.predict(dataX)

accuracy = np.sum(predict==dataY)/float(len(dataY))

print(accuracy)




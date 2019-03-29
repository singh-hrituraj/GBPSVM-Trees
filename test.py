import numpy as np 
from ensembles.RF import RF
from estimators.CART import CART


Forest=  RF(n_estimators = 2, nvartosample=2, mode='hyperplane_psvm', replace=False)


dataX = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3]])
dataY = np.array([-1,-1,-1,1,1,1])


Forest.train(dataX,dataY)
print(Forest.predict(dataX))
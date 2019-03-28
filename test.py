import numpy as np 
from ensembles import RF
from estimators.CART import CART


Tree =  CART(nvartosample=2, mode='hyperplane_psvm')


dataX = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3]])
dataY = np.array([-1,-1,-1,1,1,1])


Tree.train(dataX, dataY)

for node in Tree.nodes:
	print("DataIndx: ", node['DataIndx'] )
	print("Label: ", node['Label'])
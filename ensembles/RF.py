"""
Code by Team Parasites
Indian Institute of Technology Roorkee
"""
import numpy as np 
from estimators import CART



class RF:
	"""
	Base class for Random Forest
	"""
	def __init__(self, n_estimators = 100, minparent=2, minleaf=1, criterion='gini', replace = True):
		self.n_estimators = n_estimators
		self.minparent    = minparent
		self.minleaf      = minleaf
		self.criterion    = criterion
		self.replace      = replace


	def train(self, X, Y):
		n = len(Y)
		for i in range(self.n_estimators):
			if replace:
				TDindx = np.round((n-1)*np.random.rand(n,1))
			else:
				TDindx = np.arange(n-1)

			Random_Forest =  CART()
			Random_Forest.train()

			if i%50 == 0:
				if method=='gini':
					print(i, ' Decision Trees Generated in Random Forest with Gini Method')
					











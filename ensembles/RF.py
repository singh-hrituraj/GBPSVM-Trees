"""
Code by Team Parasites
Indian Institute of Technology Roorkee
"""
import numpy as np 
from estimators.CART import CART
from scipy import stats
np.random.seed(0)

class RF:
	"""
	Base class for Random Forest
	"""
	def __init__(self, n_estimators = 100, minparent=2, nvartosample=None, minleaf=1, mode='axis_parallel_cut', method = 'c',sss_mode='Tikhonov', replace = True):
		self.n_estimators = n_estimators
		self.minparent    = minparent
		self.minleaf      = minleaf
		self.replace      = replace
		self.method       = method
		self.mode         = mode
		self.sss_mode     = sss_mode
		self.nvartosample = nvartosample
		self.trees        = []



	def generate_tree(self, X, Y):
		if not self.nvartosample:
			self.nvartosample = X.shape[1]
		tree = CART(nvartosample=self.nvartosample, mode=self.mode, method=self.method,sss_mode=self.sss_mode, minparent=self.minparent, minleaf=self.minleaf)
		tree.train(X,Y)
		return tree

	def _add_tree(self, tree):
		self.trees.append(tree)


	def train(self, X, Y):
		n = len(Y)


		for i in range(self.n_estimators):
			if self.replace:
				indices = [int(x) for x in np.round((n-1)*np.random.rand(n,1)).squeeze()]
			else:
				indices = np.arange(n)


			tree = self.generate_tree(X[indices], Y[indices])
			self._add_tree(tree)

			# if i%5 == 0:
			# 	if self.mode=='axis_parallel_cut':
			# 		print(i, ' Axis Parallel Decision Trees Generated in Random Forest')
			# 	elif self.mode =='hyperplane_psvm':
			# 		print(i, ' Geometric Decision Trees Generated in Random Forest')

	def predict(self, X):
		
		if self.method in ['c', 'g']:
			Outputs    = [tree.predict(X) for tree in self.trees]
			Outputs    = np.array(Outputs).squeeze()
			
			Output, _  = stats.mode(Outputs)
			Output     = np.array(Output).squeeze()


		else:
			Outputs    = [tree.predict(X) for tree in self.trees]
			Outputs    = np.array(Outputs)
			Output     = np.mean(Outputs, axis=0).squeeze()


		return Output

	def print(self):
		for i in range(len(self.trees)):
			print("Tree:     ", i)
			self.trees[i].print()

					











"""
Code by Team Parasites
Indian Institute of Technology Roorkee
"""
import numpy as np 
from estimators.CART import CART
from scipy import stats



class GBT:
	"""
	Base class for gradient boosted trees
	"""


	def __init__(self, n_estimators = 50, learning_rate=0.1,  minparent=2, nvartosample=None, minleaf=1, mode='axis_parallel', method = 'c',sss_mode='Tikhonov'):
		self.n_estimators  = n_estimators
		self.minparent     = minparent
		self.nvartosample   = nvartosample
		self.learning_rate = learning_rate
		self.minleaf       = minleaf
		self.mode          = mode
		self.method        = method
		self.sss_mode      = sss_mode
		self.trees         = []

	def generate_tree(self, X, Y):
		if not self.nvartosample:
			self.nvartosample = X.shape[1]
		tree = CART(nvartosample=self.nvartosample, mode=self.mode, method=self.method,sss_mode=self.sss_mode, minparent=self.minparent, minleaf=self.minleaf)
		tree.train(X,Y)
		return tree

	def _add_tree(self, tree):
		self.trees.append(tree)


	def train(self, X, Y):

		for i in range(self.n_estimators):

			tree = self.generate_tree(X, Y)
			self._add_tree(tree)


			Y   = Y - tree.predict(X)

			if i%1 == 0:
				if self.method=='axis_parallel_cut':
					print(i, ' Axis Parallel Decision Trees Generated in Gradient Boosted Method')
				elif self.method =='hyperplane_psvm':
					print(i, ' Geometric Decision Trees Generated in Gradient Boosted Method')
	def predict(self, X):
		outputs = []



		for tree in self.trees:
			output = tree.predict(X)
			outputs.append(output)


		outputs = np.array(outputs)

		outputs = np.sum(outputs, axis=0)

		return outputs




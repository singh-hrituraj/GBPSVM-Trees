import numpy as np 
from utils import *



class CART:
	"""
	Base class for classification and regression trees
	"""

	def __init__(self, criterion ='gini', method = 'c', minparent = 2, minleaf = 1, weights = None, nvartosample=None):
		self.method        = method
		self.criterion     = criterion
		self.minparent     = minparent
		self.minleaf       = minleaf
		self.weights   	   = weights
		self.nvartosample  = nvartosample
		self.nodes         = {}


	def _get_node(self):
		node = {}
		node['DataIndx']      = None
		node['CutVariable']   = 0
		node['CutValue']      = 0
		node['Flag']          = False
		node['Label']         = 0
		node['ChildNode']     = 0

		return node




	def train(self, X, Y):
		N = len(Y) #No of Samples
		L = 2*np.ceil(N/self.minleaf - 1) #Maximum number of nodes
		M = X.shape[2] #No of features per instance

		#Creating the root node
		self.nodes[0]             = self._get_node()
		self.nodes[0]['DataIndx'] = np.arange(N)
		self.nodes[0]['Flag']     = True



		if method in ['c', 'g']:
			unique_labels, inverse_indices = np.unique(Y, return_inverse=True)
			num_labels                     = len(unique_labels)
		else:
			num_labels                     = []


		current_node = 0

		while self.nodes[current_node][Flag]:
			free_node = _get_node()
			currentDataIndx = self.nodes[current_node]['DataIndx']
			



 






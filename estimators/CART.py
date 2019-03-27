import numpy as np 
from scipy import stats
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
		self.nodes         = []


	def get_node(self):
		node = {}
		node['DataIndx']      = None
		node['Label']         = 0
		node['ChildNode']     = 0
		node['Variables']     = 0
		node['Plane']         = 0

		return node

	def add_node(self, node):
		self.nodes.append(node)


	def axis_parallel_cut(self, X, Y):

		num_features            = X.shape[1]
		pre_gini                = gini_from_distribution(Y)





		cut_values   = np.zeros(num_features,)
		impurity     = np.zeros(num_features,)

		for i in range(num_features):
			cut_values[i], impurity[i] = gini_boundary(X[:,i], Y, self.minleaf)

		min_index    = np.argmin(impurity)
		min_impurity = impurity[min_index]
		bestCutValue = cut_values[min_index]

		if min_impurity < pre_gini:
			bestCutVar = min_index
		else:
			bestCutVar = -1

		if bestCutValue == np.amax(X[:,min_index]) or bestCutValue == np.amin(X[:, min_index]):
			bestCutVar = -1
		return bestCutVar, bestCutValue












	def bestCutNode(self,  X, Y, mode = 'axis_parallel_cut', delta = 0.01):

		if len(self.weights) > 0:
			Wcd = W[currentDataIndx]
		else:
			Wcd = []

		if mode == 'axis_parallel_cut':
			bestCutVar, bestCutValue = self.axis_parallel_cut(X, Y)
			num_variables            = X.shape[1]
			Plane                    = np.zeros(num_variables+1,)
			Plane[bestCutVar]        = 1
			Plane[-1]                = bestCutValue
			return Plane
		elif mode == 'hyperplane_psvm_delta':
			Plane = self.hyperplane_psvm(X,Y, delta)
			return Plane
		elif mode == 'hyperplane_psvm':
			Plane = self.hyperplane_psvm(X,Y)
			return Plane
		elif mode == 'hyperplane_psvm_subspace':
			Plane = self.hyperplane_psvm_subspace(X,Y)
			return Plane
		elif mode == 'hyperplane_pca':
			Plane = self.hyperplane_pca(X,Y)
			return Plane
		elif mode == 'hyperplane_lda':
			Plane = self.hyperplane_lda(X,Y)
			return Plane

		else:
			print ("This method ", mode, " has not been implemented yet!")
			exit()







		




	def train(self, X, Y):
		N = len(Y) #No of Samples
		L = 2*np.ceil(N/self.minleaf - 1) #Maximum number of nodes
		M = X.shape[2] #No of features per instance

		#Creating the root node
		new_node             = self._get_node()
		new_node['DataIndx'] = np.arange(N)
		self._add_node(new_node)



		if method in ['c', 'g']:#Classification
			unique_labels, inverse_indices = np.unique(Y, return_inverse=True)
			num_labels                     = len(unique_labels)
		else:#Regression
			num_labels                     = []


		current_node_idx = 0

		#While the current node is still not solved fully
		while current_node_idx<L:
			new_node_left   = self._get_node()
			new_node_right  = self._get_node()
			current_node    = self.nodes[current_node_idx]
			currentDataIndx = current_node['DataIndx']


			#If node is totally pure
			if len(np.unique(Y[currentDataIndx])) == 1:
				if method in ['c', 'g']:
					current_node['Label'] = unique_labels[0]
				else:
					current_node['Label'] = Y[currentDataIndx[0]]


			else:
				if len(currentDataIndx) >= self.minparent:

					#Selecting m random variable/attributes
					node_variables = np.random.permutation(M)
					node_variables = node_variables[0:self.nvartosample]



					bestPlane = self.bestCutNode(X[currentDataIndx, node_variables], Y[currentDataIndx])

					if bestCutVar !=-1:
						current_node['Variables']   = node_variables
						current_node['Plane']       = bestPlane
						PlaneOutput                 = np.dot(X[currentDataIndx,node_variables], bestPlane[:-1])

						new_node_left['DataIndx']   = currentDataIndx[PlaneOutput<=bestPlane[-1]
						new_node_right              = currentDataIndx[PlaneOutput > bestPlane[-1]
						current_node['ChildNode']   = len(self.nodes)

						#Adding the newly created left and right nodes to the tree
						self._add_node(new_node_left)
						self._add_node(new_node_right)

					else:
						if method in ['c', 'g']:
							[leaf_label,_]           = stats.mode(Y[currentDataIndx], axis=None)
							current_node['Label']    = leaf_label
						else:
							current_node['Label']    = np.mean(Y[currentDataIndx])
				else:
					if method in ['c', 'g']:
						[leaf_label,_]           = stats.mode(Y[currentDataIndx], axis=None)
						current_node['Label']    = leaf_label
					else:
						current_node['Label']    = np.mean(Y[currentDataIndx])

			current_node_idx += 1











			



 






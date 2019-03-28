import numpy as np 
import numpy.linalg.matrix_rank as rank
from scipy import stats
from utils.utils import *

class CART:
	"""
	Base class for classification and regression trees
	"""

	def __init__(self, criterion ='gini', method = 'c', minparent = 2, minleaf = 1, weights = None, nvartosample=None, mode='axis_parallel_cut'):
		self.method        = method
		self.criterion     = criterion
		self.minparent     = minparent
		self.minleaf       = minleaf
		self.weights   	   = weights
		self.nvartosample  = nvartosample
		self.nodes         = []
		self.mode          = mode


	def _print_tree(self):
		#To be Implemented
		pass


	def _get_node(self):
		node = {}
		node['DataIndx']      = None
		node['Label']         = -1
		node['leftNode']      = -1
		node['rightNode']     = -1
		node['Variables']     = -1
		node['Plane']         = -1

		return node

	def _add_node(self, node):
		self.nodes.append(node)




	def axis_parallel_cut(self, X, Y, variables):
		

		num_features            = X.shape[1]
		pre_gini                = gini_from_distribution(Y)





		cut_values   = np.zeros(num_features,)
		impurity     = np.zeros(num_features,)

		for i in range(num_features):
			cut_values[i], impurity[i] = gini_boundary(X[:,i], Y, self.minleaf)
			print(impurity[i])
			print(cut_values[i])

		min_index    = np.argmin(impurity)
		min_impurity = impurity[min_index]
		bestCutValue = cut_values[min_index]

		if min_impurity < pre_gini:
			bestCutVar = variables[min_index]
		else:
			bestCutVar = -1

		if bestCutValue == np.amax(X[:,min_index]) or bestCutValue == np.amin(X[:, min_index]):
			bestCutVar = -1
		return bestCutVar, bestCutValue


	def hyperplane_psvm(self, X, Y, variables, sss_mode='Tikhonov',delta=None):
		labels    = np.unique(Y)
 
		if len(labels)==1:
			splitFlag  = -1
			Plane      = None
			return splitFlag, Plane

		if len(labels)>=3:
			group1, group2 = group(X, Y)
		else:
			group1, group2 = labels


		m1 = len(group1)
		m2 = len(group2)

		indicesGroup1 = np.isin(Y, group1)
		indicesGroup2 = np.isin(Y, group2)

		YGroupA = Y[indicesGroup1]
		YGroupB = Y[indicesGroup2]

		XGroupA = X[indicesGroup1,:]
		XGroupB = X[indicesGroup2,:]


		onesA   = np.ones((len(XGroupA),1))
		onesB   = np.ones((len(XGroupB),1))

		A       = np.concatenate(XGroupA, onesA, axis=1) 
		B       = np.concatenate(XGroupB, onesB, axis=1)


		G       = np.dot(A.T, A)/len(A)
		H       = np.dot(B.T, B)/len(B)
		rankDG  = rank(G) == len(G)
		rankDH  = rank(H) == len(H)

		if  rankDG and rankDH:
			eigenValues, eigenVectors = GeneralizedEigenSoln(G, H)#Implemented in Utils

			W1 = eigenVectors[:, 0]
			W2 = eigenVectors[:,-1]
			W  = np.stack(W1,W2, axis=1)



		else:
			if sss_mode   == 'Tikhonov':
				if rankDG:
					G  = G + delta*np.eye(len(G))
				if rankDH:
					H  = H + delta*np.eye(len(H))
				eigenValues, eigenVectors = GeneralizedEigenSoln(G, H)#Implemented in Utils

				W1 = eigenVectors[:, 0]
				W2 = eigenVectors[:,-1]
				W  = np.stack(W1,W2, axis=1)




				#To be Implemented
			elif sss_mode == 'axisParallel':
				bestCutVar, bestCutValue = axis_parallel_cut(X,Y,variables)

				W1      = np.zeros(len(G),1)  
				W1[-1]  = bestCutValue

				W       = np.stack(W1, W1, axis=1)

		splitFlag, Plane = selectHyperplane(X,Y,W, minleaf=self.minleaf)#To be Implemented

		return splitFlag, Plane








	def bestCutNode(self,  X, Y, variables, mode = 'axis_parallel_cut', delta = 0.01):


		if mode == 'axis_parallel_cut':
			bestCutVar, bestCutValue = self.axis_parallel_cut(X, Y, variables=variables)
			num_variables            = X.shape[1]
			Plane                    = np.zeros(num_variables+1,)
			Plane[bestCutVar]        = 1
			Plane[-1]                = bestCutValue
			return Plane, bestCutVar
		elif mode == 'hyperplane_psvm_delta':
			splitFlag, Plane = self.hyperplane_psvm(X,Y, delta)
			return Plane, splitFlag
		elif mode == 'hyperplane_psvm':
			splitFlag, Plane = self.hyperplane_psvm(X,Y)
			return Plane, splitFlag
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
		M = X.shape[1] #No of features per instance

		#Creating the root node
		new_node             = self._get_node()
		new_node['DataIndx'] = np.arange(N)
		self._add_node(new_node)



		if self.method in ['c', 'g']:#Classification
			unique_labels, inverse_indices = np.unique(Y, return_inverse=True)
			num_labels                     = len(unique_labels)
		else:#Regression
			num_labels                     = []


		current_node_idx = 0

		#While the current node is still not solved fully
		while current_node_idx<len(self.nodes):
			new_node_left   = self._get_node()
			new_node_right  = self._get_node()
			current_node    = self.nodes[current_node_idx]
			currentDataIndx = current_node['DataIndx']


			#If node is totally pure
			if len(np.unique(Y[currentDataIndx])) == 1:
				if self.method in ['c', 'g']:
					current_node['Label'] = [Y[currentDataIndx[0]]]
				else:
					current_node['Label'] = [Y[currentDataIndx[0]]]


			else:
				if len(currentDataIndx) >= self.minparent:

					#Selecting m random variable/attributes
					# node_variables = np.random.permutation(M)
					# node_variables = node_variables[0:self.nvartosample]
					node_variables   = np.arange(M)




					bestPlane, bestCutVar = self.bestCutNode(X[currentDataIndx][:,node_variables], Y[currentDataIndx], mode=self.mode, variables=node_variables)


					if bestCutVar !=-1:
						current_node['Variables']   = node_variables
						current_node['Plane']       = bestPlane
						PlaneOutput                 = np.dot(X[currentDataIndx][:,node_variables], bestPlane[:-1])

						new_node_left['DataIndx']   = currentDataIndx[PlaneOutput<=bestPlane[-1]]
						new_node_right['DataIndx']  = currentDataIndx[PlaneOutput > bestPlane[-1]]
						current_node['leftNode']    = len(self.nodes)
						current_node['rightNode']   = len(self.nodes)+1

						#Adding the newly created left and right nodes to the tree
						self._add_node(new_node_left)
						self._add_node(new_node_right)

					else:
						if self.method in ['c', 'g']:
							leaf_label,_             = stats.mode(Y[currentDataIndx], axis=None)

						else:
							current_node['Label']    = np.mean(Y[currentDataIndx])
				else:
					if self.method in ['c', 'g']:
						[leaf_label,_]           = stats.mode(Y[currentDataIndx], axis=None)
						current_node['Label']    = leaf_label
					else:
						current_node['Label']    = np.mean(Y[currentDataIndx])

			current_node_idx += 1


	def predict(self, X):
		M = len(X)

		output           = []
		current_node     = self.nodes[0]

		for i in range(M):
			x = X[i,:]

			while current_node['leftNode']!=-1 and current_node['rightNode']!=-1 :
				direction = np.dot(current_node['Plane'][:-1], x[current_node['Variables']])
				if direction <=current_node['Plane'][-1]:
					current_node = self.nodes[current_node['leftNode']]
				else:
					current_node = self.nodes[current_node['rightNode']]


			output.append(current_node['Label'])

		return np.array(output)


















			



 






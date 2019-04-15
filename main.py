import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
from ensembles.GBT import GBT
from ensembles.RF import RF
from config import parse_config
from sklearn.ensemble import GradientBoostingClassifier as GBC


def load_data(path, clean=True, fill=False):
	"""
	Loads the cleveland data from the original file
	[parameters]
	path  : path to the .csv file
	clean : drops the instances with missing attributes values if set to True, default True
	fill  : fill  the instances with missing attributes values if set to True, default False
	"""

	data = pd.read_csv(path, header=None).drop(0, axis=1)
	# data[13] = data[13].replace(2,1)
	# data[13] = data[13].replace(3,1)
	# data[13] = data[13].replace(4,1)



	data = data.replace('?', np.nan)
	data = data.dropna(0)
	print("Data Loaded")


	return np.array(data)

def PSO(parameters, X, Y, mode, method, epochs=20, c_1=2, c_2=2,  ):
	"""
	Performs Particle Swarm Optimization by optimizing the function 
	on the parameters and returns the best set of parameters.
	"""
	print("Performing PSO...")
	#Creating the particles and helper arrays
	num_particles   = len(parameters)
	num_dimensions  = len(parameters[0]+2) #Additional two for Pbest and Gbest
	particles       = np.zeros((num_particles, num_dimensions))
	velocity        = np.zeros((num_particles, num_dimensions))
	pbest_dimension = np.zeros((num_particles, num_dimensions))
	gbest_dimension = np.zeros((num_dimensions))
	p_best          = np.zeros((num_particles,1))-1000
	g_best          = -1000


	#Initializing the particles
	for idx in range(num_particles):
		particles[idx,:] = parameters[idx]

	#Performing the optimization
	for epoch in tqdm(range(epochs)):
		w = 0.9 - 0.8*(epoch/epochs) #Variable momentum weight

		#Compute fitness for each particle
		for idx in range(num_particles):
			fitness_ = fitness(particles[idx], X=X, Y=Y, mode=mode, method=method)

			if fitness_ > p_best[idx]:
				p_best[idx]          = fitness_
				pbest_dimension[idx] = particles[idx]


			if fitness_ > g_best:
				g_best          = fitness_
				gbest_dimension = particles[idx]

	

		r_1 = np.random.random()
		r_2 = np.random.random()

		velocity  = w*velocity + c_1*r_1*(p_best - particles)  + c_2*r_2*(g_best - particles)
		particles = particles + velocity


		#Keeping values within range
		for idx in range(num_particles):
			particles[idx, 0] = max(1, particles[idx, 0])
			particles[idx, 1] = max(3, particles[idx, 1])
			particles[idx, 2] = max(2, particles[idx, 2]) 




	return gbest_dimension








def fitness(parameters, X, Y, mode, method):
	"""
	Finds the fitness of a particle with a particular set of parameters
	"""


	n_estimators = int(parameters[0])
	nvartosample = int(parameters[1])
	minparent    = int(parameters[2])

	fitness      = cross_validation(X=X, Y=Y, n_estimators=n_estimators, nvartosample=nvartosample, minparent=minparent, mode=mode, method=method, folds=5)

	return fitness



def three_splits(data, ratio=.1, val=True):
	"""
	Splits the data into test, train, valid 
	"""

	if val:
		indices = np.arange(len(data))
		indices = np.random.permutation(indices)
		

		train   = round((1-2*ratio)*len(data))
		val     = train + round(ratio*len(data))


		train_data   = indices[:train]
		val_data     = indices[train:val]
		test_data    = indices[val:]

		train   = data[train_data]
		valid   = data[val_data]
		test    = data[test_data]
		return np.array(train, dtype=np.float), np.array(valid, dtype=np.float), np.array(test, dtype=np.float)
	else:
		indices = np.arange(len(data))
		indices = np.random.permutation(indices)
		

		train      = round((1-ratio)*len(data))
		train_data = indices[:train]
		test_data  = indices[train:]

		train = data[train_data]
		test  = data[test_data]
		return np.array(train, dtype=np.float), None, np.array(test, dtype=np.float)



	

def get_data(X, Y, folds=10):
	"""
	Prepares the data for performing K-Fold Cross Validation
	"""

	X_train = []
	Y_train = []
	X_val   = []
	Y_val   = []

	Full = np.concatenate([X,np.expand_dims(Y, axis=1)], axis = 1)
	#Shuffle
	ShuFull = np.random.permutation(Full)
	X       = ShuFull[:, :-1]
	Y       = ShuFull[:,  -1]


	no_in_fold = round(len(Y)/folds)
	indices    = np.arange(len(Y))

	index = 0
	while index < len(Y):
		start         = index
		end           = min(len(Y), index+no_in_fold)
		val_indices   = indices[start:end]
		train_indices = [idx for idx in indices if idx not in val_indices]


		X_val.append(X[val_indices, :])
		Y_val.append(Y[val_indices])


		X_train.append(X[train_indices, :])
		Y_train.append(Y[train_indices])
		index += no_in_fold
	return (X_train, Y_train, X_val, Y_val)



def cross_validation(X, Y, n_estimators, nvartosample, minparent, mode, method, folds=5):
	"""
	Performs K-Fold Cross Validation on the given model and returns the
	average cross validation accuracy for it
	"""
	X_train, Y_train, X_val, Y_val = get_data(X, Y, folds=folds)
	accuracy = 0.0

	for fold in range(folds):
		trainX = X_train[fold]
		trainY = Y_train[fold]

		valX   = X_val[fold]
		valY   = Y_val[fold]
		if method=='RF':
			model = RF(n_estimators=n_estimators, nvartosample=nvartosample, minparent=minparent, mode=mode)
		else:
			model = GBT(n_estimators=n_estimators, nvartosample=nvartosample, minparent=minparent, mode=mode)


		model.train(trainX, trainY)
		pred      = model.predict(valX)
		accuracy_ = np.sum(pred==valY)/float(len(valY))
		accuracy  += accuracy_

	return accuracy/folds


def grid_search(data, n_estimators, nvartosample, minparent, mode, method):
	"""
	Performs Grid Search with the given parameters and returns the best model while printing
	the best cross validation accuracy as well as the best set of parameters
	"""
	print("Performing Grid Search...")
	X = data[:,:-1]
	Y = data[:,-1]


	best_accuracy = 0

	for n_est in tqdm(n_estimators):
		for n_var in nvartosample:
			for minp in minparent:
				accuracy = cross_validation(X,Y, n_estimators=n_est, nvartosample=n_var, minparent=minp, mode=mode, method=method, folds=10)


				if accuracy>best_accuracy:
					best_accuracy = accuracy
					best_nest     = n_est
					best_n_var    = n_var
					best_minp     = minp
	if method=='RF':				
		model = RF(n_estimators=best_nest, nvartosample=best_n_var, minparent= best_minp, mode=mode)
	else:
		model = GBT(n_estimators=best_nest, nvartosample=best_n_var, minparent= best_minp, mode=mode)


	info  = open('grid_search.txt', 'a')



	info.write("Mode: \t\t\t"+ mode+'\n')
	info.write("Method: \t\t\t "+ method+ '\n')
	info.write("Best Accuracy: \t\t"+str(best_accuracy)+'\n')
	info.write("Best Number of Estimators: \t"+ str(best_nest)+ '\n')
	info.write("Best Number of Variables: \t"+ str(best_n_var)+ '\n')
	info.write("Best Number in Parents: \t"+ str(best_minp)+'\n')
	info.close()

	return model



def main(path):
	config = parse_config(path)

	data       = config['data']
	parameters = config['parameters']

	data_path    = data.get('path', 'data/data.csv')
	n_estimators = parameters.get('n_estimators', [100])
	nvartosample = parameters.get('nvartosample', [5])
	minparent    = parameters.get('minparent', [2])


	data = load_data(data_path)
	train, val, test = three_splits(data, val=False)

	trainX = train[:,:-1]
	trainY = train[:,-1]
	testX  = test[:,:-1]
	testY  = test[:,-1]



	methods   = ['RF', 'GBT']
	modes = ['axis_parallel_cut','hyperplane_psvm']


	parameters = []
	#Initialize parameters
	for n_est in n_estimators:
		for n_var in nvartosample:
			for minp in minparent:
				parameters.append([n_est, n_var, minp])
	parameters = np.array(parameters)




	for method in methods:
		for mode in modes:
			parameters = PSO(parameters= parameters,mode=mode, method=method,X=data[:, :-1], Y=data[:,-1], )
			info       = open('pso_optimization.txt', 'a')
			info.write("Mode: \t\t\t"+ mode+'\n')
			info.write("Method: \t\t\t "+ method+ '\n')
			info.write("Best Number of Estimators: \t"+ str(parameters[0])+ '\n')
			info.write("Best Number of Variables: \t"+ str(parameters[1])+ '\n')
			info.write("Best Number in Parents: \t"+ str(parameters[2])+'\n')
			info.close()




	# model = GBT(n_estimators=10, minparent=8, nvartosample=3)
	# model.train(trainX, trainY)
	# pred = model.predict(testX)
	# print("Testing Accuracy: \t", np.sum(pred==testY)/float(len(testY)))

	










if __name__=='__main__':
	path = sys.argv[1]
	main(path)
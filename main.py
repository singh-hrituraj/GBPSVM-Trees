import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
from ensembles.GBT import GBT
from ensembles.RF import RF
from config import parse_config

np.random.seed(0)

def load_data(path, clean=True, fill=False):
	"""
	Loads the cleveland data from the original file
	[parameters]
	path  : path to the .csv file
	clean : drops the instances with missing attributes values if set to True, default True
	fill  : fill  the instances with missing attributes values if set to True, default False
	"""

	data = pd.read_csv(path, header=None)
	data = data.replace('?', np.nan)
	data = data.dropna(0)
	print("Data Loaded")


	return np.array(data)


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



def cross_validation(X, Y, model=None, folds=5):
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
				if mode=='RF':
					model    = RF(n_estimators =n_est, nvartosample=n_var,minparent=minp, mode=method)
				else:
					model    = GBT(n_estimators =n_est, nvartosample=n_var,minparent=minp, mode=method)
				accuracy = cross_validation(X,Y, model=model, folds=10)

				if accuracy>best_accuracy:
					best_accuracy = accuracy
					best_nest     = n_est
					best_n_var    = n_var
					best_minp     = minp

	model = RF(n_estimators=best_nest, nvartosample=best_n_var, minparent= best_minp)

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


	modes   = ['RF', 'GBT']
	methods = ['axis_parallel_cut', 'hyperplane_psvm']




	for mode in modes:
		for method in methods:
			model = grid_search(train, n_estimators=n_estimators, nvartosample=nvartosample, minparent=minparent, mode=mode, method=method)
			model.train(trainX, trainY)

			
			predtrain = model.predict(trainX)
			predtest  = model.predict(testX)

			print("Training Accuracy: \t", np.sum(predtrain==trainY)/float(len(trainY)))
			print("Testing Accuracy: \t", np.sum(predtest==testY)/float(len(testY)))










if __name__=='__main__':
	path = sys.argv[1]
	main(path)
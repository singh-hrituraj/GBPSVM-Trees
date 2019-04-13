import pandas as pd
import sys
import numpy as np
from ensembles.GBT import GBT
from ensembles.RF import RF




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


def three_splits(data):
	"""
	Splits the data into test, train, valid in ratio of 80:10:10
	"""

	train = []
	valid = []
	test  = []

	for num, instance in enumerate(data):
		if num%10==0:
			valid.append(instance)
		elif num%5==0:
			test.append(instance)
		else:
			train.append(instance)


	return np.array(train, dtype=np.float), np.array(valid, dtype=np.float), np.array(test, dtype=np.float)

def main(path):

	data = load_data(path)
	train, val, test = three_splits(data)


	trainX = train[:,:-1]
	trainY = train[:,-1]

	valX   = val[:,:-1]
	valY   = val[:,-1]

	testX  = test[:,:-1]
	testY  = test[:,-1]




	forest = RF(n_estimators =200, nvartosample=4)
	forest.train(trainX, trainY)

	predtrain = forest.predict(trainX)
	predval   = forest.predict(valX)

	print("Training Accuracy: \t", np.sum(predtrain==trainY)/float(len(trainY)))
	print("Validation Accuracy: \t", np.sum(predval==valY)/float(len(valY)))









if __name__=='__main__':
	path = sys.argv[1]
	main(path)
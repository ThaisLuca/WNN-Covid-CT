
import params as params
import utils as utils
import pandas as pd

import wisardpkg as wp
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, auc

from os import path
import sys, os

def network(X, y, addressSize, ignoreZero=False, verbose=True):

	#Define model
	wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

	print("Training\n")
	# train using the input data
	wsd.train(X,y)

	# classify train data
	out = wsd.classify(X)
	out = [str(i) for i in out]

	print("Accuracy score: {}".format(accuracy_score(y, out)))
	print("F1-Score: {}".format(f1_score(y, out)))
	print("AUC: {}".format(auc(y, out)))

	#return wsd.json()



def main():

	if(not path.exists(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH)):
		# Pre-Process images and save for later
		os.makedirs(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH)
		utils.create_csv_files()

	#Load images in CSV file
	train_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH + params.TRAIN_DATASET)).sample(frac=1)

	X = train_set.drop(['target'], axis=1).values.tolist()
	
	# convert numbers to int and then to string
	#X = [[str(int(x)) for x in row] for row in X]

	Y = train_set['target'].values.tolist()
	Y = [str(y) for y in Y]
	del train_set
	print(Y[0])
	return

	addressSizes = [20] #, 25, 30, 35, 40, 45, 50]

	for addressSize in addressSizes:
		print("Current address size is {}".format(addressSize))
		network(X, Y, addressSize, False, True)

if __name__ == "__main__":
	sys.exit(main())
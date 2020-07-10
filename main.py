
import params as params
import utils as utils
import pandas as pd

import wisardpkg as wp
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, auc

from os import path
import sys, os

def main():

	if(not path.exists(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH)):
		# Pre-Process images and save for later
		os.makedirs(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH)
		utils.create_csv_files()

	#Load images in CSV file
	train_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH + params.TRAIN_DATASET, dtype=int)).sample(frac=1)
	val_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH + params.VALIDATION_DATASET, dtype=int)).sample(frac=1)

	X = train_set.drop(['target'], axis=1).values.tolist()
	X_val = val_set.drop(['target'], axis=1).values.tolist()

	y = train_set['target'].values.tolist()
	y_val = val_set['target'].values.tolist()
	y, y_val = [str(i) for i in y], [str(y_v) for y_v in y_val]
	del train_set, val_set

	addressSizes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 64]

	for addressSize in addressSizes:
		print("Current address size is {}\n".format(addressSize))
		
		#Define model
		wsd = wp.Wisard(addressSize, ignoreZero=False, verbose=True)

		# train using the input data
		wsd.train(X,y)

		# classify train data
		out = wsd.classify(X)
		out = [str(i) for i in out]

		print("Accuracy score: {}".format(accuracy_score(y, out)))
		out_f1 = [int(i) for i in out]
		y_f1 = [int(i) for i in y]
		print("F1-Score: {}".format(f1_score(y_f1, out_f1)))
		#print("AUC: {}".format(auc(y, out)))
		print("\n")

		out = wsd.classify(X_val)
		out = [str(i) for i in out]

		print("Accuracy score: {}".format(accuracy_score(y_val, out)))
		out_f1 = [int(i) for i in out]
		y_f1 = [int(i) for i in y_val]
		print("F1-Score: {}".format(f1_score(y_f1, out_f1)))
		#print("AUC: {}".format(auc(y_val, out)))
		print("\n")

if __name__ == "__main__":
	sys.exit(main())

import params as params
import utils as utils
import pandas as pd

import wisardpkg as wp
import numpy as np
from sklearn.metrics import accuracy_score

from os import path
import sys, os

def network(train, validation, test, addressSize, verbose=True, ignoreZero=False):
	best_model = 0
	best_model_accuracy = 0

	x_train, x_validation, x_test = train['data'], validation['data'], test['data']
	y_train, y_validation, y_test = train['target'], validation['target'], test['target']

	#Define model
	wsd = wp.Wisard(addressSize=addressSize, ignoreZero=ignoreZero, verbose=verbose)

	print("Training\n")
	# train using the input data
	wsd.train(X,y)

	# classify train data
	out_train = wsd.classify(x_train)

	print("Validation\n")
	# classify validation data
	out_val = wsd.classify(x_validation)

	#classify test data
	print("Test\n")
	out_test = wsd.classify(x_test)

	return accuracy_score(y_train, out_train), accuracy_score(y_val, out_val), accuracy_score(y_test, out_test)



def main():

	if(not path.exists(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH)):
		# Pre-Process images and save for later
		os.makedirs(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH)
		utils.create_csv_files()

	#Load images in CSV file
	train_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_FOLDER_PATH + params.TRAIN_DATASET)).sample(frac=1)
	print(train_set[train_set['target'] == 1].shape, train_set[train_set['target'] == 0].shape)

	#Define model
	wsd = wp.Wisard(20, ignoreZero=True, verbose=False)
	X = preprocess(train_set.drop(['target'], axis=1), 125).values.tolist()
	Y = train_set['target'].values.tolist()
	Y = [str(y) for y in Y]
	

	print("Training\n")
	# train using the input data
	wsd.train(X,Y)

	# classify train data
	out_train = wsd.classify(X)

	print(accuracy_score(Y, out_train))
		

def preprocess(df, threshold):
	columns = df.columns
	for column in columns:
		df[column] = np.where(df[column] >= threshold, 1, 0)
	return df

	#img = cv2.imread(a[0])
	#median = cv2.medianBlur(img, 5)
	#compare = np.concatenate((img, median), axis=1)

	#cv2.imshow('img', compare)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows

	addressSizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

	for addressSize in addressSizes:
		print("Current address size is {}".format(addressSize))
		network(addressSize)

if __name__ == "__main__":
	sys.exit(main())

import params as params
import utils as utils
import pandas as pd

import wisardpkg as wp
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from os import path
import sys, os


def network(X, y, X_test, y_test, addressSize):
	n_splits = 10

	train_accuracy_score, validation_accuracy_score, test_accuracy_score = [], [], []
	train_f1_score, validation_f1_score, test_f1_score = [], [], []
	train_auc_score, validation_auc_score, test_auc_score = [], [], []

	#Define model
	wsd = wp.Wisard(addressSize, ignoreZero=False, verbose=True)

	kf = KFold(n_splits=n_splits, shuffle=True)
	fold = 1
	for train_index, val_index in kf.split(X):
		print("FOLD:", fold)
		print("TRAIN: {} VALIDATION: {}".format(len(train_index), len(val_index)))
		X_train, X_val = [X[index] for index in train_index], [X[index] for index in val_index]
		y_train, y_val = [str(y[index]) for index in train_index], [str(y[index]) for index in val_index]

		# train using the input data
		print("Training")
		wsd.train(X_train,y_train)

		# classify train data
		print("Train data classification")
		out_train = wsd.classify(X_train)
		out_train = [str(i) for i in out_train]

		out_train = [str(i) for i in out_train]
		train_accuracy_score.append(accuracy_score(y_train, out_train))

		out_int = [int(i) for i in out_train]
		y_int = [int(i) for i in y_train]
		train_f1_score.append(f1_score(y_int, out_int))

		train_auc_score.append(roc_auc_score(y_int, out_int))

		# classify validation data
		print("Validation data classification")
		out_val = wsd.classify(X_val)
		out_val = [str(i) for i in out_val]

		validation_accuracy_score.append(accuracy_score(y_val, out_val))

		out_int = [int(i) for i in out_val]
		y_int = [int(i) for i in y_val]
		validation_f1_score.append(f1_score(y_int, out_int))

		validation_auc_score.append(roc_auc_score(y_int, out_int))

		#classify test data
		print("Test data classification")
		out_test = wsd.classify(X_test)
		out_test = [str(i) for i in out_test]

		test_accuracy_score.append(accuracy_score(y_test, out_test))

		out_int = [int(i) for i in out_test]
		y_int = [int(i) for i in y_test]
		test_f1_score.append(f1_score(y_int, out_int))

		test_auc_score.append(roc_auc_score(y_int, out_int))
		
		fold += 1
		print('\n')

	utils.save_metrics(addressSize, train_accuracy_score, validation_accuracy_score, test_accuracy_score, filename='accuracy.csv')
	utils.save_metrics(addressSize, train_f1_score, validation_f1_score, test_f1_score, filename='f1.csv')
	utils.save_metrics(addressSize, train_auc_score, validation_auc_score, test_auc_score, filename='auc.csv')

def main():

	if(str(sys.argv[1]).lower() == params.OTSU_THRESHOLD):
		if(not path.exists(os.getcwd() + params.PRE_PROCESSED_OTSU_FOLDER_PATH)):
			# Pre-Process images and save for later
			os.makedirs(os.getcwd() + params.PRE_PROCESSED_OTSU_FOLDER_PATH)
			utils.create_csv_files(pre_processing_technique=params.OTSU_THRESHOLD)

		#Load images in CSV file
		train_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_OTSU_FOLDER_PATH + params.TRAIN_DATASET, dtype=int)).sample(frac=1)
		val_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_OTSU_FOLDER_PATH + params.VALIDATION_DATASET, dtype=int)).sample(frac=1)
		test_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_OTSU_FOLDER_PATH + params.TEST_DATASET, dtype=int)).sample(frac=1)

		train_set = pd.concat([train_set, val_set])
		del val_set

	elif(str(sys.argv[1]).lower() == params.CANNY_DETECTOR):
		if(not path.exists(os.getcwd() + params.PRE_PROCESSED_CANNY_FOLDER_PATH)):
			# Pre-Process images and save for later
			os.makedirs(os.getcwd() + params.PRE_PROCESSED_CANNY_FOLDER_PATH)
			utils.create_csv_files(pre_processing_technique=params.CANNY_DETECTOR)

		#Load images in CSV file
		train_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_CANNY_FOLDER_PATH + params.TRAIN_DATASET, dtype=int)).sample(frac=1)
		val_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_CANNY_FOLDER_PATH + params.VALIDATION_DATASET, dtype=int)).sample(frac=1)
		test_set = (pd.read_csv(os.getcwd() + params.PRE_PROCESSED_CANNY_FOLDER_PATH + params.TEST_DATASET, dtype=int)).sample(frac=1)

		train_set = pd.concat([train_set, val_set])
		del val_set


	elif(str(sys.argv[1]).lower() == params.EMBEDDING_VGG16):

		# VGG-16
		train_set = (pd.read_csv(os.getcwd() + '/' + params.VGG_16_TRAINING, dtype=int)).sample(frac=1)
		test_set = (pd.read_csv(os.getcwd()  + '/' + params.VGG_16_TEST, dtype=int)).sample(frac=1)

	elif(str(sys.argv[1]).lower() == params.EMBEDDING_VGG19):

		# VGG-19
		train_set = (pd.read_csv(os.getcwd() + '/' + params.VGG_19_TRAINING, dtype=int)).sample(frac=1)
		test_set = (pd.read_csv(os.getcwd()  + '/' + params.VGG_19_TEST, dtype=int)).sample(frac=1)

	elif(str(sys.argv[1]).lower() == params.EMBEDDING_INCEPTION):

		# Inception V3
		train_set = (pd.read_csv(os.getcwd() + '/' + params.INCEPTION_V3_TRAINING, dtype=int)).sample(frac=1)
		test_set = (pd.read_csv(os.getcwd()  + '/' + params.INCEPTION_V3_TEST, dtype=int)).sample(frac=1)

	else:
		print("Error. Please choose {} or {} for pre-processing Otsu Thresholding and Canny Edge Detector techniques.".format(params.OTSU_THRESHOLD, params.CANNY_DETECTOR, params.EMBEDDING))
		print("Use {}, {} or {} for VGG-16, VGG-19 or Inception V3 embeddings.".format(params.EMBEDDING_VGG16, params.EMBEDDING_VGG19, params.EMBEDDING_INCEPTION))
		return

	X = train_set.drop(['target'], axis=1).values.tolist()
	X_test = test_set.drop(['target'], axis=1).values.tolist()

	y = train_set['target'].values.tolist()
	y_test = test_set['target'].values.tolist()
	y, y_test = [str(i) for i in y], [str(y_t) for y_t in y_test]
	del train_set, test_set

	addressSizes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 64]

	for addressSize in addressSizes:
		print("Current address size is {}\n".format(addressSize))
		network(X, y, X_test, y_test, addressSize)
		
		

if __name__ == "__main__":
	sys.exit(main())
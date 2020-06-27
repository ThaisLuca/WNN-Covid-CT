
import params as params
import utils as utils

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

		#Training dataset pre-process
		print('Training data set')
		train_set = utils.get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
		train_set += utils.get_files_path(file=params.NON_COVID_TRAINING_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
		print('Pre-processing all images..')
		images_processed = utils.pre_process_images(train_set)
		print("{} images processed".format(len(images_processed)))
		print('Saving images..')
		utils.save_images_as_csv(images_processed, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.TRAIN_DATASET) 


		#Validation dataset pre-process
		print('Validation data set')
		val_set = utils.get_files_path(file=params.COVID_VALIDATION_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
		val_set += utils.get_files_path(file=params.NON_COVID_VALIDATION_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
		print('Pre-processing all images..')
		images_processed = utils.pre_process_images(val_set)
		print("{} images processed".format(len(images_processed)))
		print('Saving images..')
		utils.save_images_as_csv(images_processed, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.VALIDATION_DATASET)

		#Test dataset pre-process
		print('Test data set')
		test_set = utils.get_files_path(file=params.COVID_TEST_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
		test_set += utils.get_files_path(file=params.NON_COVID_TEST_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
		print('Pre-processing all images..')
		images_processed = utils.pre_process_images(val_set)
		print("{} images processed".format(len(images_processed)))
		print('Saving images..')
		utils.save_images_as_csv(images_processed, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.TEST_DATASET) 
	return

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
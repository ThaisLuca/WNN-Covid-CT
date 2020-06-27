
import params as params
import pandas as pd
import numpy as np
import sys, os
import cv2

def get_files_path(file, path):
	with open(os.getcwd() + file) as f:
		content = f.readlines()
		paths = [os.getcwd() + path + x.strip() for x in content]
	return paths

def load_all_images_grayscale(images_path):
	images = []
	for path in images_path:
		images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
	return images

def apply_median_blur(images):
	blury_images = []
	for image in images:
		blury_images.append(cv2.medianBlur(image, 5))
	return blury_images

def apply_gaussian_blur(images):
	blury_images = []
	for image in images:
		blury_images.append(cv2.GaussianBlur(image, (5,5), 0))
	return blury_images

def pre_process_images(images_path):
	return apply_gaussian_blur(apply_median_blur(load_all_images_grayscale(images_path)))

def save_images_as_csv(images, n_true_labels, path, filename):
	data, target = [], []
	label = 1
	df= pd.DataFrame()
	for image in images:
		img_array = (image.flatten())
		img_array = img_array.reshape(-1,1).T
		data.append(img_array)
		target.append(label)

		if(len(data) == n_true_labels): label = 0 
	df['data'] = data
	df['label'] = target

	df.to_csv(os.getcwd() + path + filename, index=False)
	print('Images saved as CSV\n')

def create_csv_files():

	#Training dataset pre-process
	print('Training data set')
	train_set = get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(train_set)
	
	train_set += get_files_path(file=params.NON_COVID_TRAINING_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	print('Pre-processing all images..')
	images_processed = pre_process_images(train_set)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.TRAIN_DATASET) 


	#Validation dataset pre-process
	print('Validation data set')
	val_set = get_files_path(file=params.COVID_VALIDATION_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(val_set)

	val_set += get_files_path(file=params.NON_COVID_VALIDATION_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	print('Pre-processing all images..')
	images_processed = pre_process_images(val_set)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.VALIDATION_DATASET)

	#Test dataset pre-process
	print('Test data set')
	test_set = get_files_path(file=params.COVID_TEST_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(test_set)

	test_set += get_files_path(file=params.NON_COVID_TEST_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	print('Pre-processing all images..')
	images_processed = pre_process_images(val_set)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.TEST_DATASET)

def visualize_filters(image):
	img = cv2.imread(image)
	median = cv2.GaussianBlur(cv2.medianBlur(img, 5), (5,5), 0)
	compare = np.concatenate((img, median), axis=1)

	cv2.imshow('img', compare)
	cv2.waitKey(0)
	cv2.destroyAllWindows

	return

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
		image = cv2.resize(cv2.imread(path), params.DIM, interpolation=cv2.INTER_NEAREST)
		images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
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
	#return load_all_images_grayscale(images_path)
	return apply_gaussian_blur(apply_median_blur(load_all_images_grayscale(images_path)))

def apply_threshold(df, threshold):
	columns = df.columns
	for column in columns:
		df[column] = np.where(df[column] >= threshold, 1, 0)
	return df

def create_row(array, label, keys):
	dict_ = {}
	for px, key in zip(array, keys):
		dict_[key] = px
	dict_['target'] = label
	return dict_ 

def save_images_as_csv(images, n_true_labels, path, filename):
	columns = [i for i in range(params.DIM[0] * params.DIM[1])]
	columns.append('target')

	data = pd.DataFrame([], columns=columns)
	label = 1
	for image in images:
		img_array = (image.flatten())
		img_array = img_array.reshape(-1,1).T
		row = create_row(img_array[0], label, columns)
		data = data.append(row, ignore_index=True)

		if(data.shape[0] == n_true_labels): label = 0 

	data = apply_threshold(data, params.THRESHOLD)
	data.to_csv(os.getcwd() + path + filename, index=False)
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
	#compare = np.concatenate((img, median), axis=1)
	median = cv2.resize(median, params.DIM, interpolation=cv2.INTER_NEAREST)

	cv2.imshow('img', median)
	cv2.waitKey(0)
	cv2.destroyAllWindows

	return

import params as params
import pandas as pd
import numpy as np
import sys, os
import cv2

#import skimage.io
#from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

import matplotlib
import matplotlib.pyplot as plt

def visualize_filters_skimage_filters(images_path):
	image = skimage.io.imread(fname=images_path)
	image = skimage.color.rgb2gray(image)
	binary_global = image > threshold_otsu(image)

	window_size = 25
	thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
	thresh_savoula = threshold_sauvola(image, window_size=window_size)

	binary_niblack = image > threshold_niblack
	binary_savoula = image > threshold_sauvola

	plt.figure(figsize=(8,7))
	plt.subplot(2,2,1)
	plt.imshow(image, cmap=plt.cm.gray)
	plt.title('Original')
	plt.axis('off')

	plt.subplot(2,2,2)
	plt.title('Global Threshold')
	plt.imshow(binary_global, cmap=plt.cm.gray)
	plt.axis('off')

	plt.subplot(2,2,3)
	plt.imshow(binary_niblack, cmap=plt.cm.gray)
	plt.title('Niblack Threshold')
	plt.axis('off')

	plt.subplot(2,2,4)
	plt.imshow(binary_savoula, cmap=plt.cm.gray)
	plt.title('Savoula Threshold')
	plt.axis('off')

	plt.show()


def get_files_path(file, path):
	with open(os.getcwd() + file) as f:
		content = f.readlines()
		paths = [os.getcwd() + path + x.strip() for x in content]
	return paths

def load_all_images(images_path):
	images = []
	for path in images_path:
		images.append(cv2.imread(path))
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

def pre_process_images(images, pre_processing_technique):
	pp_images = []
	n_images = len(images)
	for image in images:
		image = cv2.cvtColor(cv2.resize(image, params.DIM, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
		if(pre_processing_technique == params.OTSU_THRESHOLD):
			ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		else:
			thresh1 = cv2.Canny(image, 100, 200)
		pp_images.append(thresh1)	
	return pp_images

def apply_threshold(df):
	columns = df.columns
	for column in columns:
		if(column == 'target'): continue
		df[column] = np.where(df[column] == 255, '1', '0')
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
	print("{} columns created.".format(len(columns)))
	n_images = len(images)

	data = pd.DataFrame([], columns=columns)
	label = '1'
	for image in images:
		img_array = (image.flatten())
		img_array = img_array.reshape(-1,1).T
		row = create_row(img_array[0], label, columns)
		data = data.append(row, ignore_index=True)
		print("{}\{} saved to CSV file".format(data.shape[0], n_images))

		if(data.shape[0] == n_true_labels): label = '0'

	data = apply_threshold(data)
	data.to_csv(os.getcwd() + path + filename, index=False)
	print('Images saved as CSV\n')

def create_csv_files(pre_processing_technique):

	folder = params.PRE_PROCESSED_OTSU_FOLDER_PATH if pre_processing_technique == params.OTSU_THRESHOLD else params.PRE_PROCESSED_CANNY_FOLDER_PATH

	#Training dataset pre-process
	print('Training data set')
	train_set = get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(train_set)
	
	train_set += get_files_path(file=params.NON_COVID_TRAINING_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	images = load_all_images(train_set)
	print('Pre-processing all images..')
	images_processed = pre_process_images(images, pre_processing_technique)
	print("{} images processed".format(images_processed))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=folder, filename=params.TRAIN_DATASET) 

	#Validation dataset pre-process
	print('Validation data set')
	val_set = get_files_path(file=params.COVID_VALIDATION_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(val_set)

	val_set += get_files_path(file=params.NON_COVID_VALIDATION_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	images = load_all_images(val_set)
	print('Pre-processing all images..')
	images_processed = pre_process_images(images, pre_processing_technique)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=folder, filename=params.VALIDATION_DATASET)

	#Test dataset pre-process
	print('Test data set')
	test_set = get_files_path(file=params.COVID_TEST_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(test_set)

	test_set += get_files_path(file=params.NON_COVID_TEST_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	images = load_all_images(test_set)
	print('Pre-processing all images..')
	images_processed = pre_process_images(images, pre_processing_technique)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=folder, filename=params.TEST_DATASET)

def save_metrics(addressSize, train_score, validation_score, test_score, filename):

	train_score_mean = np.mean(train_score, axis=0)
	validation_score_mean = np.mean(validation_score, axis=0)
	test_score_mean = np.mean(test_score, axis=0)

	train_score_std = np.std(train_score)
	validation_score_std = np.std(validation_score)
	test_score_std = np.std(test_score)

	matrix = {}
	matrix['addressSize'] = [addressSize]
	matrix['train_mean'] = [train_score_mean]
	matrix['validation_mean'] = [validation_score_mean]
	matrix['test_mean'] = [test_score_mean]
	matrix['train_std'] = [train_score_std]
	matrix['validation_std'] = [validation_score_std]
	matrix['test__std'] = [test_score_std]

	result = pd.DataFrame(matrix)
	with open(filename, 'a') as file:
		result.to_csv(file, index=False, header=True)


def visualize_filters_opencv_filters(image_path):
	img = cv2.imread(image_path)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#ret, thresh1 = cv2.adaptiveThreshold(img, 0, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2)
	#ret, thresh2 = cv2.adaptiveThreshold(img, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
	ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	edges_1 = cv2.Canny(img, 100, 200)
	edges_2 = cv2.Canny(img, 300, 200)

	plt.figure(figsize=(5,5))
	plt.subplot(2,2,1)
	plt.imshow(img, cmap=plt.cm.gray)
	plt.title('Original')
	plt.axis('off')

	plt.subplot(2,2,2)
	plt.title('Otsu Threshold')
	plt.imshow(thresh1, cmap=plt.cm.gray)
	plt.axis('off')

	plt.subplot(2,2,3)
	plt.title('Canny Edge Detector 100x200')
	plt.imshow(edges_1, cmap=plt.cm.gray)
	plt.axis('off')

	plt.subplot(2,2,4)
	plt.title('Canny Edge Detector 300x200')
	plt.imshow(edges_2, cmap=plt.cm.gray)
	plt.axis('off')

	plt.show()

	return

#visualize_filters_opencv_filters('resources/foto.png')
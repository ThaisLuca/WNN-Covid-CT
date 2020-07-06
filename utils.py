
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

def pre_process_images(images):
	pp_images = []
	n_images = len(images)
	for image in images:
		image = cv2.cvtColor(cv2.resize(image, params.DIM, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
		ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

def create_csv_files():

	#Training dataset pre-process
	print('Training data set')
	train_set = get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(train_set)
	
	train_set += get_files_path(file=params.NON_COVID_TRAINING_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	images = load_all_images(train_set)
	print('Pre-processing all images..')
	images_processed = pre_process_images(images)
	print("{} images processed".format(images_processed))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.TRAIN_DATASET) 

	#Validation dataset pre-process
	print('Validation data set')
	val_set = get_files_path(file=params.COVID_VALIDATION_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(val_set)

	val_set += get_files_path(file=params.NON_COVID_VALIDATION_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	images = load_all_images(val_set)
	print('Pre-processing all images..')
	images_processed = pre_process_images(images)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.VALIDATION_DATASET)

	#Test dataset pre-process
	print('Test data set')
	test_set = get_files_path(file=params.COVID_TEST_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	n_true_values = len(test_set)

	test_set += get_files_path(file=params.NON_COVID_TEST_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	images = load_all_images(test_set)
	print('Pre-processing all images..')
	images_processed = pre_process_images(images)
	print("{} images processed".format(len(images_processed)))
	print('Saving images..')
	save_images_as_csv(images_processed, n_true_values, path=params.PRE_PROCESSED_FOLDER_PATH, filename=params.TEST_DATASET)

def visualize_filters(image_path):
	img = cv2.imread(image_path)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	ret, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	compare = np.concatenate((thresh1, thresh2), axis=1)

	cv2.imshow('img', compare)
	cv2.waitKey(0)
	cv2.destroyAllWindows

	return

#visualize_filters('resources/foto.png')
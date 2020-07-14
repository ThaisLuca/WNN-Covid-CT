
import numpy as np
import pandas as pd

import params as params
import utils as utils

import tensorflow as tf
import tensorflow_hub as hub

# VGG-16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

# VGG-19
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

#InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input

import matplotlib.pyplot as plt

from keras.layers import Embedding
from keras.models import Sequential,Model

from keras.utils import to_categorical

from keras.preprocessing import image

from os import path
import sys, os
import cv2

def get_train_test_images(width, height):
	#Load images
	train_path = utils.get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	train_target = [1] * len(train_path)

	train_path += utils.get_files_path(file=params.NON_COVID_TRAINING_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	train_target += [0] * (len(train_path) - len(train_target))

	train_data = []
	for image_path in train_path:
		img = image.load_img(image_path, target_size=(width, height))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		train_data.append(img)

	del train_path

	val_path = utils.get_files_path(file=params.COVID_VALIDATION_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	val_target = [1] * len(val_path)

	val_path += utils.get_files_path(file=params.NON_COVID_VALIDATION_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	val_target += [0] * (len(val_path) - len(val_target))

	val_data = []
	for image_path in val_path:
		img = image.load_img(image_path, target_size=(width, height))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		val_data.append(img)

	train_data += val_data
	train_target += val_target
	del val_data, val_target, val_path

	test_path = utils.get_files_path(file=params.COVID_TEST_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	test_target = [1] * len(test_path)

	test_path += utils.get_files_path(file=params.NON_COVID_TEST_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	test_target += [0] * (len(test_path) - len(test_target))
	
	test_data = []
	for image_path in test_path:
		img = image.load_img(image_path, target_size=(width, height))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		test_data.append(img)

	print("Treinamento: ", len(train_data), len(train_target))
	print("Teste: ", len(test_data), len(test_target))

	return train_data, train_target, test_data, test_target

def to_csv(images, targets, dim0, dim1, filename):
	columns = [i for i in range(dim0 * dim1)]
	columns.append('target')
	print("{} columns created.".format(len(columns)))
	n_images = len(images)

	data = pd.DataFrame([], columns=columns)
	for image, target in zip(images, targets):
		img_array = (image.flatten())
		img_array = img_array.reshape(-1,1).T
		row = utils.create_row(img_array[0], target, columns)
		data = data.append(row, ignore_index=True)
		print("{}\{} saved to CSV file".format(data.shape[0], n_images))

	data.to_csv(filename, index=False)
	print('Images saved as CSV\n')


def use_vgg16(train, test):
	#Load pre-trained model
	model = VGG16(weights='imagenet', include_top=False)

	train_features, test_features = [], []
	for image in train:
		img = vgg16_preprocess_input(image)
		vgg16_feature_train = model.predict(image)
		vgg16_feature_train = np.array(vgg16_feature_train)
		train_features.append(vgg16_feature_train)
	
	for image in test:
		img = vgg16_preprocess_input(image)
		vgg16_feature_test = model.predict(image)
		vgg16_feature_test = np.array(vgg16_feature_test)
		test_features.append(vgg16_feature_test)

	return train_features, test_features

def use_vgg19(train, test):
	#Load pre-trained model
	model = VGG19(weights='imagenet', include_top=False)

	train_features, test_features = [], []
	for image in train:
		img = vgg19_preprocess_input(image)
		vgg19_feature_train = model.predict(image)
		vgg19_feature_train = np.array(vgg19_feature_train)
		train_features.append(vgg19_feature_train)
	
	for image in test:
		img = vgg19_preprocess_input(image)
		vgg19_feature_test = model.predict(image)
		vgg19_feature_test = np.array(vgg19_feature_test)
		test_features.append(vgg19_feature_test)

	return train_features, test_features

def use_inceptionV3(train, test):
	#Load pre-trained model
	model = InceptionV3(weights='imagenet', include_top=False)

	train_features, test_features = [], []
	for image in train:
		img = inception_v3_preprocess_input(image)
		inception_v3_feature_train = model.predict(image)
		inception_v3_feature_train = np.array(inception_v3_feature_train)
		train_features.append(inception_v3_feature_train)
	
	for image in test:
		img = inception_v3_preprocess_input(image)
		inception_v3_feature_test = model.predict(image)
		inception_v3_feature_test = np.array(inception_v3_feature_test)
		test_features.append(inception_v3_feature_test)

	return train_features, test_features



def main():
	print("\n\n")

	train_data, train_target, test_data, test_target = get_train_test_images(224,224)

	# Feature extraction using VGG-16
	vgg16_train_features, vgg16_test_features = use_vgg16(train_data, test_data)
	to_csv(vgg16_train_features, train_target, 224, 224, 'vgg-16-train-features.csv')
	to_csv(vgg16_test_features, test_target, 224, 224, 'vgg-16-test-features.csv')
	del vgg16_train_features, vgg16_test_features

	# Feature extraction using VGG-19
	vgg19_train_features, vgg19_test_features = use_vgg19(train_data, test_data)
	to_csv(vgg19_train_features, train_target, 224, 224, 'vgg-19-train-features.csv')
	to_csv(vgg19_test_features, test_target, 224, 224, 'vgg-19-test-features.csv')
	del vgg19_train_features, vgg19_test_features

	# Feature extraction using Inception V3
	inception_train_features, inception_test_features = use_inceptionV3(train_data, test_data)
	to_csv(inception_train_features, train_target, 224, 224, 'inception-v3-train-features.csv')
	to_csv(inception_test_features, test_target, 224, 224, 'inception-v3-test-features.csv')
	del inception_train_features, inception_test_features


if __name__ == "__main__":
	sys.exit(main())

#model = Sequential()
#model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch,
# input_length).
# the largest integer (i.e. word index) in the input should be no larger
# than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch
# dimension.

#input_array = np.random.randint(1000, size=(32, 10))

#model.compile('rmsprop', 'mse')
#output_array = model.predict(input_array)
#print(output_array[:20])
#assert output_array.shape == (32, 10, 64)

import numpy as np
import pandas as pd

import params as params
import utils as utils

import tensorflow as tf
import tensorflow_hub as hub

from keras.layers import Embedding
from keras.models import Sequential,Model

from os import path
import sys, os

def get_train_test_images():
	#Load images
	train_data = utils.get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	train_target = [1] * len(train_data)

	train_data += utils.get_files_path(file=params.NON_COVID_TRAINING_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	train_target += [0] * (len(train_data) - len(train_target))

	train_data = utils.load_all_images(train_data)

	val_data = utils.get_files_path(file=params.COVID_VALIDATION_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	val_target = [1] * len(val_data)

	val_data += utils.get_files_path(file=params.NON_COVID_VALIDATION_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	val_target += [0] * (len(val_data) - len(val_target))

	val_data = utils.load_all_images(val_data)
	train_data += val_data
	train_target += val_target
	del val_data, val_target

	test_data = utils.get_files_path(file=params.COVID_TEST_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)
	test_target = [1] * len(test_data)

	test_data += utils.get_files_path(file=params.NON_COVID_TEST_DATASET, path=params.NON_COVID_IMAGES_PROCESSED_PATH)
	test_target += [0] * (len(test_data) - len(test_target))
	test_data = utils.load_all_images(test_data)

	print("Treinamento: ", len(train_data), len(train_target))
	print("Teste: ", len(test_data), len(test_target))

	return train_data, train_target, test_data, test_target

def to_csv(images, targets, dim0, dim1):
	columns = [i for i in range(dim0 * dim1)]
	columns.append('target')
	print("{} columns created.".format(len(columns)))
	n_images = len(images)

	data = pd.DataFrame([], columns=columns)
	for image, target in zip(images, targets):
		ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		img_array = (thresh1.flatten())
		img_array = img_array.reshape(-1,1).T
		row = utils.create_row(img_array[0], target, columns)
		data = data.append(row, ignore_index=True)
		print("{}\{} saved to CSV file".format(data.shape[0], n_images))

	data = utils.apply_threshold(data)
	data.to_csv(os.getcwd() + path + filename, index=False)
	print('Images saved as CSV\n')


def use_inceptionV3(train, test):
	module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
	# images is a tensor of [batch, 299, 299, 3]
	# outputs is a tensor of [batch, 2048]
	train_outputs = module(train)
	test_outputs = module(teste)



def main():

	path = utils.get_files_path(file=params.COVID_TRAINING_DATASET, path=params.COVID_IMAGES_PROCESSED_PATH)[0]
	utils.visualize_filters_opencv_filters(path)
	return

	train_data, train_target, test_data, test_target = get_train_test_images()

	# Using InceptionV3
	train_data_features, test_data_features = use_inceptionV3(train_data, test_data)


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
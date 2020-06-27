
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

def save_images_as_csv(images, path, filename):
	for image in images:
		img_array = (image.flatten())
		img_array = img_array.reshape(-1,1).T

		with open(os.getcwd() + path + filename, 'ab') as file:
			np.savetxt(file, img_array, delimiter=",")
	print('Images saved as CSV\n')

def visualize_filters(image):
	img = cv2.imread(image)
	median = cv2.GaussianBlur(cv2.medianBlur(img, 5), (5,5), 0)
	compare = np.concatenate((img, median), axis=1)

	cv2.imshow('img', compare)
	cv2.waitKey(0)
	cv2.destroyAllWindows

	return
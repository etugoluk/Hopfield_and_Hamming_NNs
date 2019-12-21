import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(x, label=''):
	fig, ax = plt.subplots()
	ax.matshow(x.reshape((28, 28)), cmap='gray')
	fig.suptitle(label)
	plt.show()

def show_history(history, label='', foldername=''):
	fig, ax = plt.subplots(1, len(history), figsize=(10, 5))
	for i,x in enumerate(history):
		ax[i].matshow(x.reshape((28, 28)), cmap='gray')
	fig.suptitle(label)
	plt.savefig('{}/{}'.format(foldername, label))
	plt.show()

def get_images(folder):
	X = []
	labels = []
	images = glob.glob(folder + '/*.jpg')
	for img in images:
		x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
		x = x.flatten()
		x = x.astype(np.float32)
		x /= 255
		x = np.sign(x * 2 - 1)
		X.append(x)
		labels.append(int(img[6]))
	return X, labels
import numpy as np
import img_util
import noise

#initializing ang learning
X, labels = img_util.get_images('train')
N = len(X[0])
M = len(X)
W1 = np.zeros((N, M), 'float')
T = np.zeros((M), 'float')
for k, x in enumerate(X):
	for i in range(N):
		W1[i, k] = x[i] / 2
	T[k] = N / 2
eps = 0.05
W2 = np.full((M, M), eps)
for i in range(M):
	W2[i, i] = 1

#testing
Y = noise.add(X, 0.3)
for it, y in enumerate(Y):
	history = []
	history.append(y)
	y1 = np.zeros(M)
	for j in range(M):
		summ = 0
		for i in range(N):
			summ += W1[i, j] * y[i]
		y1[j] = summ + T[j]
	y2 = np.copy(y1)
	for step in range(5):
		y2_old = np.copy(y2)
		print (y2)
		s = np.zeros(M)
		for j in range(M):
			s[j] = y2_old[j] - eps * (np.sum(y2_old) - y2_old[j])
			if (s[j] <= 0):
				y2[j] = 0
			elif (s[j] <= N / 2):
				y2[j] = s[j]
			else:
				y2[j] = N / 2
		if np.array_equal(y2_old, y2) == True:
			break
	# print ('\n\n\n')
	res = np.argmax(y2)
	history.append(X[res])
	img_util.show_history(history, label=labels[it], foldername='results_HammingNN')

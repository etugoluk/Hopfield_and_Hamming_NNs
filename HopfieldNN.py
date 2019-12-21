import numpy as np
import img_util
import noise

#learning
X, labels = img_util.get_images('train')
N = len(X[0])
M = len(X)
W = np.zeros((N, N), 'float')
for x in X:
	W += np.outer(x, x)
for i in range(N):
	W[i, i] = 0
# W /= N * N * N
# print (W)

#testing
# Y = get_images('test')
Y = noise.add(X, 0.2)
isFind = True
steps = 5
error_threshold = 0.1 * N * N
for it, y in enumerate(Y):
	history = []
	history.append(y)
	min_error_count = N * N
	min_error_x = -1
	for l in range(steps):
		Z = []
		for j in range(N):
			d = 0
			for i in range(N):
				d += W[i, j] * y[i]
			Z.append(np.sign(d).astype('int'))
		history.append(np.asarray(Z))
		if np.array_equal(y, Z) == True:
			break
		y = Z.copy()
		for x in X:
			error_count = 0
			isFind = True
			for i in range(N):
				if x[i] != Z[i]:
					error_count += 1
			if error_count < min_error_count:
				min_error_count = error_count
				min_error_x = x
		if min_error_count > error_threshold:
			isFind = False
		if isFind:
			print("Image found!")
			history.append(np.asarray(min_error_x))
			img_util.show_history(history, label=labels[it], foldername="results_HopfieldNN")
			break
	if not isFind:
		print("Image not found!")

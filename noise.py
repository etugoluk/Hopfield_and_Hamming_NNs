import numpy as np

def add(train, error_rate):
	test = np.copy(train)
	for t in test:
		s = np.random.binomial(1, error_rate, len(t))
		for j in range(len(t)):
			if s[j] != 0:
				t[j] *= -1
	return test

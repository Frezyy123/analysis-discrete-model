import numpy as np

PROJECT_PATH = 'C:\Users\Alexandr\Desktop\lab_1'

with open(PROJECT_PATH+'\input.txt') as file:
	data =file.readlines()
	X = np.fromstring(data[0], sep=',')
	U = np.fromstring(data[1], sep=',')
	N = np.fromstring(data[2], sep=',')
	Y = np.fromstring(data[3], sep=',')

	
A = np.random.random(size=(X.size, X.size))
B = np.random.random(size=(X.size, U.size))
E = np.random.random(size=(X.size, N.size))
C = np.random.random(size=(Y.size, X.size))
D = np.random.random(size=(Y.size, U.size))
F = np.random.random(size=(Y.size, N.size))




X = np.matmul(A, X) + np.matmul(B, U) + np.matmul(E, N)
Y = np.matmul(C, X) + np.matmul(D, U) + np.matmul(F, N)

print('state:', X)
print('output:', Y)
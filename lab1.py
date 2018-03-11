import numpy as np


# PROJECT_PATH = 'C:\Users\Alexandr\Desktop\lab_1'
def read_data():
    with open('./input.txt') as text_data:
        data = text_data.readlines()
        X = np.fromstring(data[0], sep=',')
        U = np.fromstring(data[1], sep=',')
        N = np.fromstring(data[2], sep=',')
        Y = np.fromstring(data[3], sep=',')
        T = np.fromstring(data[4], sep=',')
    return X, U, N, Y, T


def init_coeffs():
    A = np.random.random(size=(X.size, X.size))
    B = np.random.random(size=(X.size, U.size))
    E = np.random.random(size=(X.size, N.size))
    C = np.random.random(size=(Y.size, X.size))
    D = np.random.random(size=(Y.size, U.size))
    F = np.random.random(size=(Y.size, N.size))
    return A, B, C, D, E, F


states = []
outputs = []


def calculate_model(A, B, C, D, E, F, X, U, N, Y, T):
    states.append(X)
    outputs.append(Y)
    for i in range(T):
        X = np.matmul(A, X) + np.matmul(B, U) + np.matmul(E, N)
        Y = np.matmul(C, X) + np.matmul(D, U) + np.matmul(F, N)
        states.append(X)
        outputs.append(Y)
    return

def stability(A):
    poles = np.linalg.eig(A)[0]
    abs_poles = np.absolute(poles)
    if np.any(abs_poles >1):
        print("System is not stable")
    elif np.any(abs_poles ==1):
        print("System na granice stability")
    else:
        print("System is stable")
    return poles


X, U, N, Y, T = read_data()
A, B, C, D, E, F = init_coeffs()

calculate_model(A, B, C, D, E, F, X, U, N, Y, T)


stability(A)
print('state:', states)
print('output:', outputs)

import numpy as np
import matplotlib.pyplot as plt


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
    # A = np.random.random(size=(X.size, X.size))
    A = [[-0.03333333, 0.1, 0.01666667, 0.2, 0.26666667, 0.02],
         [0.33333333, 0.03333333, -0.33333333, 0.33333333, 0.06666667, 0.33333333],
         [-0.33333333, -0.33333333, -0.2, 0.33333333, 0.1, 0.16666667],
         [0.02333333, -0.03, 0.33333333, 0.06666667, -0.16666667, -0.26666667],
         [0.33333333, -0.33333333, 0.23333333, 0.3, -0.3, -0.1],
         [0.33333333, 0.33333333, 0.26666667, 0.16666667, 0.13333333, 0.06666667]]

    B = np.random.random(size=(X.size, U.size))
    E = np.random.random(size=(X.size, N.size))
    C = np.random.random(size=(Y.size, X.size))
    D = np.random.random(size=(Y.size, U.size))
    F = 20*np.random.random(size=(Y.size, N.size))
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
    stable = 0
    if np.any(abs_poles > 1):
        print("System is not stable")
    elif np.any(abs_poles == 1):
        stable = 1
        print("System na granice stability")
    else:
        print("System is stable")
        stable = 2
    return stable


def quality_criteria(stable, outputs):
    end_value = outputs[-1]
    if stable == 2:
        sigma = np.max(outputs) / end_value

def characteristics(X,Y):
    sigma = 0
    regulation_time = 0
    state_velue = 0
    for j in range(Y.size):
        X0 = X_old
        outputs = []

        for s in range(10000):
            X0 = np.matmul(A, X0) + np.matmul(B, U) + np.matmul(E, N)
            Y0 = np.matmul(C, X0) + np.matmul(D, U) + np.matmul(F, N)
            outputs.append(Y0[j])

        outputs = np.array(outputs)
        state_velue = outputs[-1]
        sigma = ((np.max(outputs) - state_velue) / state_velue) * 100.
        for index, out in enumerate(outputs):
            regulation_time = index
            if abs((out - state_velue) / float(state_velue)) < 0.05:
                break

    return sigma, regulation_time, state_velue

def plot_model(outputs, T):
    plt.figure()
    times = [i for i in range(T)]
    outputs = outputs[1:int(T[0] + 2)]
    plt.plot(times, outputs)
    plt.show()

def plot_model_filtered(filtered, outputs, T, sigma , regulation_time, end_value):
    plt.figure(1)

    times = [i for i in range(T)]
    outputs = outputs[0:int(T[0])]

    plt.plot(times, outputs)

    times = [i for i in range(T)]
    filtered = filtered[0:int(T[0])]
    plt.title('red - with filter blue w/o filter \n'
              'sigma = {0}, regulation_time = {1}, end_value = {2}'
              .format(str(round(sigma,3)), str(regulation_time), str(round(end_value,3))))

    plt.plot(times, filtered)
    plt.show()

def filter(Y):
    window = 3
    alpha = 2 / float((window + 1))

    Ny = len(Y)
    EMAy = [0]
    EMAy[0] = Y[0]
    for i in range(1, Ny):
        EMAy.append(alpha * Y[i] + (1 - alpha) * EMAy[i - 1])

    return EMAy


X, U, N, Y, T = read_data()
A, B, C, D, E, F = init_coeffs()
#
X_old = list(X)
calculate_model(A, B, C, D, E, F, X, U, N, Y, T)

#plot_model(outputs, T)

filtered_Y = filter(outputs)
stability(A)
sigma, regulation_time, end_value = characteristics(X,Y)
plot_model_filtered(filtered_Y, outputs, T,sigma, regulation_time, end_value)

#print(A)

#print('state:', states)
#print('output:', outputs)

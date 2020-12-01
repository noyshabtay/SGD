#################################
# Your name: Noy Shabtay
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper_hinge():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']

	neg, pos = "0", "8"
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = (labels[train_idx[:6000]] == pos)*2-1

	validation_data_unscaled = data[train_idx[6000:], :].astype(float)
	validation_labels = (labels[train_idx[6000:]] == pos)*2-1

	test_data_unscaled = data[60000+test_idx, :].astype(float)
	test_labels = (labels[60000+test_idx] == pos)*2-1

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def helper_ce():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']
	
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = labels[train_idx[:6000]]

	validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
	validation_labels = labels[train_idx[6000:8000]]

	test_data_unscaled = data[8000+test_idx, :].astype(float)
	test_labels = labels[8000+test_idx]

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD_hinge(data, labels, C, eta_0, T):
    """
	Implements Hinge loss using SGD.
	"""
    n = len(data)
    w = np.zeros(len(data[0]))
    for t in range(1, T + 1):
        eta_t = eta_0 / t
        i = np.random.randint(0, n)
        w = update_w(w, data[i], labels[i], eta_t, C)
    return w


def SGD_ce(data, labels, eta_0, T):
    """
	Implements multi-class cross entropy loss using SGD.
	"""
    w = np.zeros((10, len(data[0])))
    for t in range(1, T + 1):
        i = np.random.randint(0, len(data))
        gradients = calculate_gradients(w, data[i], labels[i])
        for j in range(10):
            gradients[j] = np.multiply(eta_0*(-1), gradients[j])
            w[j] += gradients[j]
    return w


#################################
############# q1 ################
def update_w(w, x, y, eta_t, C):
    updated_w = w
    if np.dot(np.multiply(y, w), x) < 1:
        updated_w = np.add(np.multiply((1 - eta_t), w), np.multiply(eta_t*C*y, x))
    else:
        updated_w = np.multiply((1 - eta_t), w)
    return updated_w

def calculate_accuracy(data, labels, w):
    errors = 0
    n = len(data)
    for i in range(n):
        y = labels[i]
        x = data[i]
        sign = np.dot(w, x)
        predicted_y = 1 if sign >= 0 else -1
        if y != predicted_y:
            errors += 1
    return 1 - (errors/n)


def q1_a(train_data, train_labels, validation_data, validation_labels): #accuracy as func of eta_0
    avg_accuracy = []
    etas = []
    for pow in np.arange(-5, 6):
        eta_0 = np.float_power(10, pow)
        print("power is ", pow, "and eta is ", eta_0)
        sum_accuracy = 0.0
        for i in range(10):
            print(i)
            w = SGD_hinge(train_data, train_labels, 1, eta_0, 1000)
            accu = calculate_accuracy(validation_data, validation_labels, w)
            sum_accuracy += accu
        avg = sum_accuracy / 10
        avg_accuracy.append(avg)
        etas.append(eta_0)

    plt.plot(etas, avg_accuracy, label='Average Accuracy')
    plt.ylabel('Average Accuracy')
    plt.xlabel('eta')
    plt.xscale('log')
    plt.savefig('q1_a.png')
    print(etas) 
    print(avg_accuracy)


def q1_b(train_data, train_labels, validation_data, validation_labels, BEST_ETA): #accuracy as func of C.
    avg_accuracy = []
    c_s = []
    for pow in np.arange(-5, 5):
        c = np.float_power(10, pow)
        print("power is ", pow, "and C is ", c)
        sum_accuracy = 0.0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, BEST_ETA, 1000)
            accu = calculate_accuracy(validation_data, validation_labels, w)
            sum_accuracy += accu
        avg = sum_accuracy / 10
        avg_accuracy.append(avg)
        c_s.append(c)

    plt.plot(c_s, avg_accuracy, label = 'Average Accuracy')
    plt.ylabel('Average Accuracy')
    plt.xlabel('C')
    plt.xscale('log')
    plt.savefig('q1_b.png')
    print(c_s)
    print(avg_accuracy)


def q1_c_d(train_data, train_labels,test_data, test_labels, BEST_ETA, BEST_C):
    w = SGD_hinge(train_data, train_labels, BEST_C, BEST_ETA, 20000)
    w_reshaped = np.reshape(w, (28, 28))
    plt.imshow(w_reshaped, interpolation='nearest')
    plt.savefig('q1_c.png')
    accuracy = calculate_accuracy(test_data, test_labels, w)
    print(accuracy)

############# q2 ################

def calculate_soft_max(w, x):
    # e^wj*xi / sum(e^(wi*xi))
    dot_products = [np.dot(x, w[i]) for i in range(10)]
    max_wi_xi = np.max(dot_products)
    dot_products =  dot_products - max_wi_xi
    exp = np.exp(dot_products)
    soft_max_all = exp / np.sum(exp)
    return soft_max_all


def calculate_gradients(w, x, y):
    # lce(w,x,y) = (p(i|x,y)-I{i=y})*x
    soft_max_all = calculate_soft_max(w, x)
    label = int(y)
    soft_max_all[label] = soft_max_all[label] - 1
    gradients = []
    for i in range(10):
      gradients.append(soft_max_all[i] * x)
    return gradients


def calculate_accuracy_ce(w, data, labels):
    errors = 0
    n = len(data)
    for i in range(n):
        dot_products = [np.dot(data[i], w[j]) for j in range(10)]
        max_index = np.argmax(dot_products)
        y = int(labels[i])
        if(max_index != y):
           errors += 1
    return 1 - (errors/n)


def q2_a(train_data, train_labels, validation_data, validation_labels):
    etas = [np.float_power(10, k) for k in np.arange(-7,-5.5, 0.1)]
    avg_accuracy = []
    for eta in etas:
        sum_accuracy = 0.0
        for i in range(10):
            w = SGD_ce(train_data, train_labels, eta, 1000)
            accuracy = calculate_accuracy_ce(w, validation_data, validation_labels)
            sum_accuracy += accuracy
        avg = sum_accuracy / 10
        avg_accuracy.append(avg)  

    plt.plot(etas, avg_accuracy)
    plt.xlabel('eta')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.savefig('q2_a.png')
    print(etas)
    print(avg_accuracy)

def q2_b(train_data, train_labels, b_BEST_ETA):
    w = SGD_ce(train_data, train_labels, b_BEST_ETA, 20000)
    fig = plt.figure()
    for i in range(1,11):
      fig.add_subplot(2, 5, i)
      plt.imshow(np.reshape(w[i-1], (28, 28)), interpolation='nearest')
    plt.savefig('q2_b.png')

    
def q2_3(train_data, train_labels,test_data, test_labels, b_BEST_ETA):
    classifiers = SGD_ce(train_data, train_labels, b_BEST_ETA, 20000)
    print(calculate_accuracy_ce(classifiers, test_data, test_labels))

if __name__ == "__main__":
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    q1_a(train_data, train_labels, validation_data, validation_labels)
    BEST_ETA = 0.816
    q1_b(train_data, train_labels, validation_data, validation_labels, BEST_ETA)
    BEST_C = 0.000144
    q1_c_d(train_data, train_labels,test_data, test_labels, BEST_ETA, BEST_C)
    #accuracy on test = 0.9928352098259979

    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    q2_a(train_data, train_labels, validation_data, validation_labels)
    b_BEST_ETA = 6.309573444801892e-07 #10**(-6.2)
    q2_b(train_data, train_labels, b_BEST_ETA)
    q2_3(train_data, train_labels,test_data, test_labels, b_BEST_ETA)
    #accuracy on test = 0.8835

#################################
# hw7.py
import jax.numpy as jnp
from jax import grad
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt
datapath = "./"
np.random.seed(0)


#################### Task 1 ###################

def model(x, w):
    # option 1: stack 1
    f = x
    # print("before stack 1, x.shape: ", f.shape)
    # tack a 1 onto the top of each input point all at once
    o = jnp.ones((1, np.shape(f)[1]))
    f = jnp.vstack((o,f))
    # print("after stack 1, the X.shape:", f.shape)
    # compute linear combination and return
    a = jnp.dot(f.T,w)
    # option 2:
    #a = w[0, :] + jnp.dot(x.T, w[1:, :])
    return a.T


# multi-class softmax cost function
def multiclass_softmax(w, x_p, y_p):

	# pre-compute predictions on all points
	all_evals = model(x_p,w)
	# print(f"all_evals[:, 0:5].T={all_evals[:, 0:5].T}")

	# logsumexp trick
	maxes = jnp.max(all_evals, axis=0)
	a = maxes + jnp.log(jnp.sum(jnp.exp(all_evals - maxes), axis=0))

	# compute cost in compact form using numpy broadcasting
	b = all_evals[y_p.astype(int).flatten(), jnp.arange(np.size(y_p))]
	cost = jnp.sum(a - b)

	# return average
	return cost/float(np.size(y_p))

def gradient_descent_auto(g,max_its,w,x,y,diminishing_alpha=False,alpha = 1):
    gradient = grad(g)
    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    cost_history = [g(w,x,y)] # container for corresponding cost function history
    acc_history = [accuracy(w,x,y)]
    for k in range(1, max_its+1):
        if diminishing_alpha:
            alpha = 1/float(k)
        else:
            alpha = alpha
        ascent_direction = gradient(w,x,y)
        w = w - alpha * ascent_direction
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w,x,y))
        acc_history.append(accuracy(w,x,y))

    return weight_history,cost_history,acc_history

def accuracy(w,x,y):
    y_pred = np.argmax(model(x,np.array(w)),axis=0)
    correct = np.size(y) - jnp.count_nonzero(y_pred-y)
    return correct/jnp.size(y)


# standard normalization function
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust

    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds

    # create inverse standard normalizer
    inverse_normalizer = lambda data: data*x_stds + x_means

    # return normalizer
    return normalizer,inverse_normalizer

# compute eigendecomposition of data covariance matrix for PCA transformation
def PCA(x):
    # regularization parameter for numerical stability
    lam = 10**(-7)

    # create the correlation matrix
    P = float(x.shape[1])
    Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

    # use numpy function to compute eigenvalues / vectors of correlation matrix
    d,V = np.linalg.eigh(Cov)
    return d,V

# PCA-sphereing - use PCA to normalize input features
def PCA_sphereing(x):
    # Step 1: mean-center the data
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_centered = x - x_means

    # Step 2: compute pca transform on mean-centered data
    d,V = PCA(x_centered)

    # Step 3: divide off standard deviation of each (transformed) input,
    # which are equal to the returned eigenvalues in 'd'.
    stds = (d[:,np.newaxis])**(0.5)

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((stds.shape))
        adjust[ind] = 1.0
        stds += adjust

    normalizer = lambda data: np.dot(V.T,data - x_means)/stds

    # create inverse normalizer
    inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

    # return normalizer
    return normalizer,inverse_normalizer

def plot_cost_history(cost_history,name):
    plt.figure()
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='.')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.title(name)
    plt.savefig(name + ".png")
    plt.show()

def compare_cost_history(cost1, cost2, cost3):
    plt.figure()
    plt.plot(np.arange(1, len(cost1) + 1), cost1, 'k-', marker=".", label="mnist cost")
    plt.plot(np.arange(1, len(cost2) + 1), cost2, "r-", marker='.', label="Standard normalization cost")
    plt.plot(np.arange(1, len(cost3) + 1), cost3, "g-", marker='.', label="pca sphereing cost")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig("CostHistoryComparison.png")
    plt.show()

def compare_accuracy_history(cost1, cost2, cost3):
    plt.figure()
    plt.plot(np.arange(1, len(cost1) + 1), cost1, 'k-', marker=".", label="mnist accuracy")
    plt.plot(np.arange(1, len(cost2) + 1), cost2, "r-", marker='.', label="Standard normalization accuracy")
    plt.plot(np.arange(1, len(cost3) + 1), cost3, "g-", marker='.', label="pca sphereing accuracy")
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("AccuracyComparison.png")
    plt.show()

def run_task1():
    # import MNIST
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    #print(x)
    # re-shape input/output data
    x = x.T
    y = np.array([int(v) for v in y])[np.newaxis,:]

    x_Task1 = x.to_numpy()
    x_Task1 = x_Task1[:,0:50000]

    y = y[:,0:50000]

    #print(x_Task1)
    #print(np.shape(x_Task1))
    #print(np.shape(x)) # (784, 70000)
    #print(np.shape(y)) # (1, 70000)
    standard_normaliser, standard_denormalizer = standard_normalizer(x_Task1)
    pca_normalizer,pca_denormalizer = PCA_sphereing(x_Task1)

    x_stdN = standard_normaliser(x_Task1)
    x_pcaS = pca_normalizer(x_Task1)

    #print(x_stdN)
    #print(x_pcaS)

    alpha_mnist = 10 ** -5
    alpha_std = 1
    alpha_pca = 10

    w_mnist = np.random.randn(785,10)* 10**-3
    #print(w_mnist)

    weight_history_mnist, cost_history_mnist, accuracy_history_mnist = gradient_descent_auto(g=multiclass_softmax,max_its=10,w=w_mnist,x=x_Task1,y=y,diminishing_alpha=False,alpha = alpha_mnist)
    weight_history_std, cost_history_std, accuracy_history_std = gradient_descent_auto(multiclass_softmax,10,w_mnist,x_stdN,y,False,alpha =alpha_std)
    weight_history_pca, cost_history_pca, accuracy_history_pca = gradient_descent_auto(multiclass_softmax,10,w_mnist,x_pcaS,y,False,alpha =alpha_pca)

    plot_cost_history(cost_history_mnist,'mnist')
    plot_cost_history(cost_history_std,'std')
    plot_cost_history(cost_history_pca,'pca')

    compare_cost_history(cost_history_mnist,cost_history_std,cost_history_pca)
    compare_accuracy_history(accuracy_history_mnist,accuracy_history_std,accuracy_history_pca)

##################
def lsc(w,x,y):
    return jnp.sum((model(x,w)-y)**2)

def l1(w,x,y,lamda):
    return (lsc(w,x,y)+lamda*jnp.sum(jnp.absolute(w[1:])))/jnp.size(y)

def plot_histogram(w,name):
    plt.figure()
    plt.bar(jnp.arange(14),jnp.squeeze(w))
    plt.ylim([-4, 3])
    plt.title(name)
    plt.savefig(name)
    plt.show()

def gradient_descent_auto1(g,max_its,w,x,y,diminishing_alpha=False,alpha = 1,lamda=0):
	# TODO: compute gradient module using jax
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w,x,y,lamda)]          # container for corresponding cost function history
	for k in range(1, max_its+1):
		if diminishing_alpha:
			alpha = 1/float(k)
		else:
			alpha = alpha
		# TODO: evaluate the gradient, store current weights and cost function value
		ascent_direction = gradient(w,x,y,lamda)
		w = w - alpha * ascent_direction

		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w,x,y,lamda))
	return weight_history,cost_history

def run_task2():
    # load in data 
    csvname =  datapath + 'boston_housing.csv'
    data = np.loadtxt(csvname, delimiter = ',')
    x = data[:-1,:]
    y = data[-1:,:]
    print(np.shape(x))
    print(np.shape(y))
    # input shape: (13, 506)
    # output shape: (1, 506)
    normaliser,denormaliser = standard_normalizer(x)
    x_stdN = normaliser(x)

    g = lambda w,x,y,lamda: l1(w,x,y,lamda)

    lamda1= 0
    lamda2= 50
    lamda3= 100
    lamda4= 150

    w1 = np.random.randn(14,1)
    w1 = jnp.array(w1)
    
    weight_histroylam1,cost_histroylam1 = gradient_descent_auto1(g=g,x=x_stdN,y=y,w=w1,alpha=0.001,max_its=2000,lamda=0)
    weight_histroylam2,cost_histroylam2 = gradient_descent_auto1(g=g,x=x_stdN,y=y,w=w1,alpha=0.001,max_its=2000,lamda=50)
    weight_histroylam3,cost_histroylam3 = gradient_descent_auto1(g=g,x=x_stdN,y=y,w=w1,alpha=0.001,max_its=2000,lamda=100)
    weight_histroylam4,cost_histroylam4 = gradient_descent_auto1(g=g,x=x_stdN,y=y,w=w1,alpha=0.001,max_its=2000,lamda=150)

    plot_cost_history(cost_histroylam1,'Lambda is 0')
    plot_cost_history(cost_histroylam2,'Lambda is 50')
    plot_cost_history(cost_histroylam3,'Lambda is 100')
    plot_cost_history(cost_histroylam4,'Lambda is 150')

    print('final cost to show convergence(lambda = 0)')
    print(cost_histroylam1[-1])
    print('final cost to show convergence(lambda = 50)')
    print(cost_histroylam2[-1])
    print('final cost to show convergence(lambda = 100)')
    print(cost_histroylam3[-1])
    print('final cost to show convergence(lambda = 150)')
    print(cost_histroylam4[-1])
    
    plot_histogram(weight_histroylam1[-1],'lambda is 0')
    plot_histogram(weight_histroylam2[-1],'lambda is 50')
    plot_histogram(weight_histroylam3[-1],'lambda is 100')
    plot_histogram(weight_histroylam4[-1],'lambda is 150')
if __name__ == '__main__':
	run_task1()
	run_task2()



import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import grad

import matplotlib.pyplot as plt

datapath = "./"


def train_test_split(*arrays, test_size=0.2, shuffle=True, rand_seed=0):
    # set the random state if provided
    np.random.seed(rand_seed)

    # initialize the split index
    array_len = len(arrays[0].T)
    split_idx = int(array_len * (1 - test_size))

    # initialize indices to the default order
    indices = np.arange(array_len)

    # shuffle the arrays if shuffle is True
    if shuffle:
        np.random.shuffle(indices)

    # Split the arrays
    result = []
    for array in arrays:
        if shuffle:
            array = array[:, indices]
        train = array[:, :split_idx]
        test = array[:, split_idx:]
        result.extend([train, test])

    return result

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

def linear_model(x,w):
	return (w[0] + jnp.dot(x.T,w[1:])).T

def softmax(w,x,y):
    return jnp.sum(jnp.log(1 + jnp.exp(-y*linear_model(x,w))))

def perceptron(w,x,y):
	return jnp.sum(jnp.maximum(0,-y*linear_model(x,w)))

def l1(w,x,y,lamda):
    return (softmax(w,x,y)+lamda*jnp.sum(jnp.absolute(w[1:])))/jnp.size(y)

def gradient_descent_auto(g,max_its,w,x,y,diminishing_alpha=False,alpha = 1,lamda=0):
      # TODO: compute gradient module using jax
    gradient = grad(g)
    weight_history = [w]           # container for weight history
    cost_history = [g(w,x,y,lamda)] # container for corresponding cost function history
    accuracy_history = [get_accuracy(w,x,y)] # container for accuracy        
    for k in range(1, max_its+1):
        if diminishing_alpha:
            alpha = 1/float(k)
        elif k%200 == 0 and np.mean(cost_history[-100:-50]) - np.mean(cost_history[-50:]) < 1e-8:
                if alpha > 1e-3:
                    alpha = alpha * 0.1
                else:
                    print(f" early stop at k = {k}")
                    break
        else:
            alpha = alpha
        
        ascent_direction = gradient(w,x,y,lamda)
        w = w - alpha * ascent_direction
        weight_history.append(w)
        cost_history.append(g(w,x,y,lamda))
        accuracy_history.append(get_accuracy(w,x,y))
    return weight_history,cost_history,accuracy_history
      
def plot_cost_history(cost_history,name):
    plt.figure()
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='.')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.title(name)
    plt.savefig(name + ".png")
    plt.show()

def plot_histogram(w,name):
    plt.figure()
    plt.bar(jnp.arange(len(w)),jnp.squeeze(w))
    plt.ylim([-4, 3])
    plt.title(name)
    plt.savefig(name)
    plt.show()

def plot_accuracy_history(cost1):
    plt.figure()
    plt.plot(np.arange(1, len(cost1) + 1), cost1, 'k-', marker=".", label="accuracy")
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("AccuracyComparison.png")
    plt.show()
      

def kcv(x,y,num_data,test_size=0.2):
      np.random.seed(0)
      indices = np.arange(num_data)
      np.random.shuffle(indices)
      fold_size = int(num_data *  test_size)
      num_folds = int(num_data/fold_size)
      print("num folds:")
      print(num_folds)
      hyperparams=[[10**-1,10**-3],[10**-1,10**-1],[10**-2,10**-2]]
      best_average_accuracy=0
      best_hyperparam=[]
      g = lambda w,x,y,lamda: l1(w,x,y,lamda)
      for i in hyperparams:
        print("hyperparams- learning rate: {}, regularizing factor:{}".format(i[0],i[1]))
        average_validation_accuracy = 0
        for k in range(num_folds):
            validation_indices = indices[k * fold_size: (k + 1) * fold_size]
            train_indices = np.concatenate([indices[:k * fold_size], indices[(k + 1) * fold_size:]])
            # print("train indices")
            # print(train_indices)
            # print("validation indices")
            # print(validation_indices)
            x_train, y_train = x.T[train_indices], y.T[train_indices]

            x_train = x_train.T
            y_train = y_train.T
            standard_normaliser, standard_denormalizer = standard_normalizer(x_train)
            x_train = standard_normaliser(x_train)

            w = np.random.randn(7129,1) * 10**-5
            w = jnp.array(w)

            weight_histroy,cost_history,accuracy_history = gradient_descent_auto(g=g,x=x_train,y=y_train,w=w,alpha=i[0],max_its=1000,lamda=i[1])

            x_validation, y_validation = x.T[validation_indices], y.T[validation_indices]
            x_validation = x_validation.T
            y_validation = y_validation.T
            x_validation = standard_normaliser(x_validation)

            average_validation_accuracy = average_validation_accuracy + get_accuracy(weight_histroy[-1],x_validation,y_validation)           
        
        print("total validation accuracy:")
        print(average_validation_accuracy)
        average_validation_accuracy = average_validation_accuracy / num_folds
        if average_validation_accuracy > best_average_accuracy:
            best_average_accuracy = average_validation_accuracy
            best_hyperparam = i
      
      print("Best hyperparams- learning rate: {}, regularizing factor:{}".format(best_hyperparam[0],best_hyperparam[1]))
      print("Best average accuracy: " + str(best_average_accuracy))
      w_t = np.random.randn(7129,1) * 10**-5
      w_t = jnp.array(w)
      standard_normaliser1, standard_denormalizer1 = standard_normalizer(x)
      x = standard_normaliser1(x)
      print("Running Gradient descent on training set using best hyperparams")
      weight_histroy_t,cost_history_t,accuracy_history_t = gradient_descent_auto(g=g,x=x,y=y,w=w_t,alpha=best_hyperparam[0],max_its=1000,lamda=best_hyperparam[1])
      
      train_accuracy1 = get_accuracy(weight_histroy_t[-1],x,y)
      print("Training accuracy: ")
      print(train_accuracy1)

      return weight_histroy_t,cost_history_t,standard_normaliser1

def get_accuracy(w,x,y):
    y_pred = jnp.sign(w[0] + jnp.dot(x.T,w[1:])).T
    count_mis = jnp.count_nonzero(y - y_pred)
    accuracy = 1-(count_mis/float(jnp.size(y)))
    return accuracy

def run_task1():
    csvname = datapath + 'new_gene_data1.csv'
    data = np.loadtxt(csvname, delimiter=',')
    x = data[:-1, :]
    y = data[-1:, :]

    print(np.shape(x))  # (7128, 72)
    print(np.shape(y))  # (1, 72)

    np.random.seed(0)  # fix randomness
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, rand_seed=0)

    print(np.shape(x_train))
    print(np.shape(y_train))
    weight_history_final,cost_history_final,standard_normaliser2 = kcv(x_train,y_train,len(x_train.T),0.16)

    plot_cost_history(cost_history_final,"cost history")

    x_test = standard_normaliser2(x_test)
    test_accuracy = get_accuracy(weight_history_final[-1],x_test,y_test)
    print("Test accuracy is:")
    print(test_accuracy)
    
    optimal_weights=weight_history_final[-1].flatten()
    indices1 = np.argsort(np.abs(optimal_weights))[::-1] 
    print("The top 5 genes are:")
    for index in indices1[:5]:
        print(f"Genes with index {index}")


if __name__ == '__main__':
    run_task1()

import numpy as np
from jax import grad
import jax.numpy as jnp 
import matplotlib.pyplot as plt 

datapath = "./"
np.random.seed(1)

#################### Task 1 ###################
def linear_model(x,w):
	return (w[0] + jnp.dot(x.T,w[1:])).T

def softmax(w,x,y):
	return jnp.mean(jnp.log(1 + jnp.exp(-y*linear_model(x,w))))

def plot_cost_history(cost_history,name): 
    plt.figure() 
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='.')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.savefig(name + ".png") 
    plt.show()

# taken from hw2
def gradient_descent_auto(g,max_its,w,x,y,diminishing_alpha=False,alpha = 1):
	# TODO: compute gradient module using jax
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w,x,y)] # container for corresponding cost function history
	accuracy_history = [] # container for accuracy        
	for k in range(1, max_its+1):
		if diminishing_alpha:
			alpha = 1/float(k)
		else:
			alpha = alpha
		# TODO: evaluate the gradient, store current weights and cost function value
		ascent_direction = gradient(w,x,y)
		w = w - alpha * ascent_direction

		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w,x,y))

		y_pred = jnp.sign(w[0] + jnp.dot(x.T,w[1:]))
		count_mis = jnp.count_nonzero(y.T - y_pred)
		accuracy = 1-(count_mis/jnp.size(y))
		accuracy_history.append(accuracy)

	return weight_history,cost_history,accuracy_history

def run_task1(): 
	# load in data
	csvname = datapath + '2d_classification_data_v1.csv'
	data = np.loadtxt(csvname,delimiter = ',')

	# take input/output pairs from data
	x = data[:-1, :]
	y = data[-1:, :] 

	print(np.shape(x)) # (1, 11)
	print(np.shape(y)) # (1, 11)

	# TODO: fill in the rest of the code 
	w = jnp.array([jnp.float32(3.0),jnp.float32(3.0)])
	g = lambda w,x,y : softmax(w,x,y)

	weight_history, cost_history, accuracy_history = gradient_descent_auto(g,2000,w,x,y,False,alpha=1.0) 
	
	y_pred = jnp.sign(weight_history[-1][0] + weight_history[-1][1] * x)
	
	n = 1000
	x_new = np.linspace(np.min(x), np.max(x), n).reshape(1, -1)
	y_fitted = jnp.tanh(weight_history[-1][0] + weight_history[-1][1] * x_new.T)

	print("Total number of misclassifications")
	print(jnp.count_nonzero(y_pred - y))
	print("Indices misclassified")
	print(jnp.nonzero(y_pred - y))

	print("Accuracy")
	print(str((1 - (jnp.count_nonzero(y_pred - y)/jnp.size(y)))*100.0) + '%')

	#plot cost history
	plot_cost_history(cost_history, 'Cost_History_Task1')

	#plot
	plt.scatter(x.T,y.T,color='k', label = "Data")
	plt.plot(x_new.T,y_fitted,color='r',label="Tanh fitted curve")
	plt.legend()
	plt.savefig("Tanh_Task1.png")
	plt.show()


#################### Task 2 ###################

def perceptron(w,x,y):
	return jnp.mean(jnp.maximum(0,-y*linear_model(x,w)))

def compare_cost_history(costs_softmax, costs_perceptron):
	plt.figure()
	plt.plot(np.arange(1, len(costs_softmax) + 1), costs_softmax, 'k-', marker=".", label="softmax cost")
	plt.plot(np.arange(1, len(costs_perceptron) + 1), costs_perceptron, "r-", marker='.', label="perceptron cost")
	plt.xlabel("iterations")
	plt.ylabel("cost")
	plt.legend()
	plt.savefig("CostHistoryComparison.png")
	plt.show()

def compare_accuracy_history(acc_softmax, acc_perceptron):
	plt.figure()
	plt.plot(np.arange(1, len(acc_softmax) + 1), acc_softmax, 'k-',  label="softmax accuracy")
	plt.plot(np.arange(1, len(acc_perceptron) + 1), acc_perceptron, "r-",  label="perceptron accuracy")
	plt.xlabel("iterations")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig("AccuracyHistoryComparison.png")
	plt.show()

def run_task2(): 
	# data input
	csvname = datapath + 'breast_cancer_data.csv'
	data = np.loadtxt(csvname,delimiter = ',')

	# get input and output of dataset
	x = data[:-1, :]
	y = data[-1:, :] 

	print(np.shape(x)) # (8, 699)
	print(np.shape(y)) # (1, 699)
	
	# TODO: fill in the rest of the code 
	w = np.random.randn(9,1)
	print("Initial weights:")
	print(w)
	print("alpha=0.1")
	print("max_its=1945")

	g_softmax = lambda w,x,y : softmax(w,x,y)
	g_perceptron = lambda w,x,y: perceptron(w,x,y)

	weight_historyS, cost_historyS, accuracy_historyS = gradient_descent_auto(g_softmax,1945,w,x,y,False,alpha =0.1) 
	weight_historyP, cost_historyP, accuracy_historyP = gradient_descent_auto(g_perceptron,1945,w,x,y,False, alpha = 0.1) 

	y_predS = jnp.sign(weight_historyS[-1][0] + jnp.dot(x.T,weight_historyS[-1][1:]))
	y_predP = jnp.sign(weight_historyP[-1][0] + jnp.dot(x.T,weight_historyP[-1][1:]))

	count_misS = jnp.count_nonzero(y.T - y_predS)
	count_misP = jnp.count_nonzero(y.T - y_predP)

	#find indices of misclassified predictions
	ind_misS = jnp.nonzero(y.T - y_predS)
	ind_misP = jnp.nonzero(y.T - y_predP)

	#for i in ind_misP:
	#	print(y.T[i])
	#	print(y_predP[i])

	#for i in ind_misS:
	#	print(y.T[i])
	#	print(y_predS[i])

	#print(ind_misS)
	

	print("Perceptron:")
	print("Misclassifications: " + str(count_misP))
	print("Indices of Y misclassified")
	print(ind_misP)
	print("Accuracy: " +  str((1-(count_misP/jnp.size(y)))*100) + "%") 
	
	print("Softmax:")
	print("Misclassifications: " + str(count_misS))
	print("Indices of Y misclassified")
	print(ind_misS)
	print("Accuracy: " +  str((1-(count_misS/jnp.size(y)))*100) + "%") 
	
	compare_cost_history(cost_historyS,cost_historyP)
	compare_accuracy_history(accuracy_historyS, accuracy_historyP)

	

if __name__ == '__main__':
	run_task1()
	run_task2()




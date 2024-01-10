# hw3.py 

import jax.numpy as jnp 
from jax import grad 
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 
import numpy as np 
import matplotlib.pyplot as plt 

datapath="./"

#################### Task 1 ###################

def linear_model(x,w):
	return w[0] + jnp.dot(x,w[1])

def least_squares(w,x,y):
	return jnp.mean((linear_model(x,w)-y)**2)

def LSE(y_pred,y):
	return jnp.mean((y_pred-y)**2)

# taken from hw2
def gradient_descent_auto(g,max_its,w,x,y,diminishing_alpha=False,alpha = 1):
	# TODO: compute gradient module using jax
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w,x,y)]          # container for corresponding cost function history
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
	return weight_history,cost_history



def run_task1(): 
	# import the dataset
	csvname = datapath + 'student_debt_data.csv'
	data = np.loadtxt(csvname,delimiter=',')

	# extract input - for this dataset, these are times
	x = data[:,0]

	# extract output - for this dataset, these are total student debt
	y = data[:,1]

	print(np.shape(x))
	print(np.shape(y))

	# TODO: fit a linear regression model to the data  
	#min max scaling of input data
	#min_x = jnp.min(x)
	#max_x = jnp.max(x)
	#Z Score scaling used
	a = jnp.mean(x)
	d = jnp.std(x)
	scaled_x = (x - a)/d
	#print(scaled_x)
	w = jnp.array([jnp.float32(0),jnp.float32(0)])
	g = lambda w,x,y: least_squares(w,x,y)
	weight_history, cost_history = gradient_descent_auto(g,20000,w,scaled_x,y,True)

	slope = weight_history[-1][1]/d
	print('Slope of fitted line:') 
	print(slope)
	intercept = weight_history[-1][0] - (slope*a)
	print('Intercept of fitted line:') 
	print(intercept)
	print("Equation of fitted line:")
	print(str(slope) + "x + " + str(intercept))
	y_pred = lambda x: slope*x + intercept
	y_predicted  = y_pred(x)

	#plot
	plt.scatter(x,y,color='k', label = "Data")
	plt.plot(x,y_predicted,color='r',label="Least Square Fit")
	plt.xlabel("Student data")
	plt.ylabel("Target Debt")
	plt.legend()
	plt.savefig("Task1.png")
	plt.show()

	#prediction for 2030
	print('In the year 2030, target debt is:')
	print(y_pred(jnp.float32(2030)))
	



#################### Task 2 ###################

def least_absolute(w,x,y):
	return jnp.mean((jnp.abs(linear_model(x,w)-y)))

def linear_model(x,w):
	return w[0] + jnp.dot(x,w[1])

def LAE(y_pred,y):
	return jnp.mean(jnp.abs(y_pred-y))

def run_task2():
	# load in dataset
	data = np.loadtxt(datapath + 'regression_outliers.csv',delimiter = ',')
	x = data[:-1,:]
	y = data[-1:,:]

	print(np.shape(x))
	print(np.shape(y))

	# TODO: fit two linear models to the data 
	w = jnp.array([jnp.float32(0),jnp.float32(0)])

	g1 = lambda w,x,y: least_absolute(w,x,y)
	g2 = lambda w,x,y: least_squares(w,x,y)

	weight_history1, cost_history1 = gradient_descent_auto(g1,1000,w,x,y,True)
	weight_history2, cost_history2 = gradient_descent_auto(g2,1000,w,x,y,True)

	y_pred1 = lambda x: weight_history1[-1][1]*x.T + weight_history1[-1][0] 
	y_pred2 = lambda x: weight_history2[-1][1]*x.T + weight_history2[-1][0] 

	y_predicted1 = y_pred1(x)
	y_predicted2 = y_pred2(x)

	#plot
	plt.scatter(x.T,y.T,color='k', label = "Data points")
	plt.plot(x.T,y_predicted1,color='r',label = "Least Absolute Fit")
	plt.plot(x.T,y_predicted2,color='g', label = "Least Squares fit")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.legend()
	plt.savefig("Task2.png")
	plt.show()

	print("Equation of Least Square fitted line:")
	print(str(weight_history2[-1][1]) + "x + " + str(weight_history2[-1][0]))

	print("Equation of Least Absolute fitted line:")
	print(str(weight_history1[-1][1]) + "x + " + str(weight_history1[-1][0]))

if __name__ == '__main__':
	run_task1()
	run_task2() 



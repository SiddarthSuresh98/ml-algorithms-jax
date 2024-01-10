import jax.numpy as jnp
import numpy as np  
from jax import grad 
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 


import matplotlib.pyplot as plt 

#################### Task 1 ###################

def cost_func(w):
	## TODO: calculate the cost given w
	cost = None
	g = lambda w: (w**4+w**2+10*w)/float(50)
	cost = g(w)
	return cost

def gradient_func(w):
	## TODO: calculate the gradient given w
	d_g = lambda w: (2*(w**3) + w + 5)/float(25)
	return d_g(w)

def plot_cost_history(cost_history,name): 
    plt.figure() 
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='.')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.savefig(name + ".png") 
    plt.show()

def compare_cost_history1(costs1, costs2, costs3):
	plt.figure()
	plt.plot(np.arange(1, len(costs1) + 1), costs1, 'k-', marker=".", label="cost when alpha is 1")
	plt.plot(np.arange(1, len(costs2) + 1), costs2, "r-", marker='.', label="cost when alpha is 0.1")
	plt.plot(np.arange(1, len(costs2) + 1), costs3, "g-", marker='.', label="cost when alpha is 0.01")
	plt.xlabel("k")
	plt.ylabel("cost")
	plt.legend()
	plt.savefig("comparison1.png")
	plt.show()

def gradient_descent(g, gradient, alpha,max_its,w):

	# run the gradient descent loop
	cost_history = [g(w)]        # container for corresponding cost function history
	weight_history = [w]
	for k in range(1,max_its+1):       
		# TODO: evaluate the gradient, store current weights and cost function value
		ascent_direction = gradient(w)
		w = w - alpha*ascent_direction

		# collect final weights
		cost_history.append(g(w))  
		weight_history.append(w)
	return weight_history,cost_history



def run_task1(): 
	print("run task 1 ...")
	# TODO: Three seperate runs using different steplength
	g = lambda w: (w**4+w**2+10*w)/float(50)
	d_g = d_g = lambda w: (2*(w**3) + w + 5)/float(25)
	w = jnp.float64(2)
	print("Value of the cost function at w0:")
	print(cost_func(w))
	print("Value of the derivative of cost function at w0:")
	print(gradient_func(w))
	weight_history1, cost_history1 = gradient_descent(g,d_g,alpha=1,max_its=1000,w=w)
	print("alpha=1")
	print("final weight:")
	print(weight_history1[-1])
	print("final cost:")
	print(cost_history1[-1])
	plot_cost_history(cost_history1,"cost_history_alpha_1")
	weight_history2, cost_history2 = gradient_descent(g,d_g,alpha=0.1,max_its=1000,w=w)
	print("alpha=0.1")
	print("final weight:")
	print(weight_history2[-1])
	print("final cost:")
	print(cost_history2[-1])
	plot_cost_history(cost_history2,"cost_history_alpha_2")
	weight_history3, cost_history3 = gradient_descent(g,d_g,alpha=0.01,max_its=4000,w=w)
	print("alpha=0.01")
	print("final weight:")
	print(weight_history3[-1])
	print("final cost:")
	print(cost_history3[-1])
	plot_cost_history(cost_history3,"cost_history_alpha_3")
	compare_cost_history1(cost_history1,cost_history2,cost_history3)
	print("task 1 finished")



#################### Task 2 ###################

def compare_cost_history2(costs_fixed, costs_diminishing):
	plt.figure()
	plt.plot(np.arange(1, len(costs_fixed) + 1), costs_fixed, 'k-', marker=".", label="cost when alpha is fixed")
	plt.plot(np.arange(1, len(costs_diminishing) + 1), costs_diminishing, "r-", marker='.', label="cost when alpha is diminshing")
	plt.xlabel("k")
	plt.ylabel("cost")
	plt.legend()
	plt.savefig("comparison2.png")
	plt.show()

def gradient_descent_auto(g,max_its,w, diminishing_alpha=False,alpha = 1):

	# TODO: compute gradient module using jax
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w)]          # container for corresponding cost function history
	for k in range(1, max_its+1):
		if diminishing_alpha:
			alpha = 1/float(k)
		else:
			alpha = alpha
		# TODO: evaluate the gradient, store current weights and cost function value
		ascent_direction = gradient(w)
		w = w - alpha * ascent_direction

		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w))
	return weight_history,cost_history

def run_task2(): 
	print("run task 2 ...")
	# TODO: implement task 2
	#g = lambda w: w if w >= 0 else (-1*w)
	g = lambda w: jnp.abs(w)
	w = jnp.float64(2)

	weight_history_fixed,cost_history_fixed = gradient_descent_auto(g,alpha=0.5,max_its=20,diminishing_alpha=False,w=w)
	print("alpha is fixed")
	print("final weight:")
	print(weight_history_fixed[-1])
	print("Cost history when alpha is fixed:")
	print(cost_history_fixed)
	plot_cost_history(cost_history_fixed,"cost_history_fixed")

	weight_history_diminishing,cost_history_diminishing = gradient_descent_auto(g,max_its=20,diminishing_alpha=True,w=w)
	print("alpha is diminishing")
	print("final weight:")
	print(weight_history_diminishing[-1])
	print("Cost history when alpha is diminishing:")
	print(cost_history_diminishing)
	plot_cost_history(cost_history_diminishing,"cost_history_diminishing")

	compare_cost_history2(cost_history_fixed,cost_history_diminishing)

	print("task 2 finished")


if __name__ == '__main__':
	run_task1()
	run_task2()  
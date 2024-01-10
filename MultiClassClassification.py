import jax.numpy as jnp
import numpy as np
from jax import grad
import matplotlib.pyplot as plt

np.random.seed(0)
datapath = "./"

#################### Task 3 ###################


# A helper function to plot the original data
def show_dataset(x, y):
  y = y.flatten()
  num_classes = np.size(np.unique(y.flatten()))
  accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
  # initialize figure
  plt.figure()

  # color current class
  for a in range(0, num_classes):
    t = np.argwhere(y == a)
    t = t[:, 0]
    plt.scatter(
      x[0, t],
      x[1, t],
      s=50,
      color=accessible_color_cycle[a],
      edgecolor='k',
      linewidth=1.5,
      label="class:" + str(a))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(bbox_to_anchor=(1.1, 1.05))

  plt.savefig("data.png")
  plt.close()
    

def show_dataset_labels(x, y, modelf, n_axis_pts=120):
  y = y.flatten()
  num_classes = np.size(np.unique(y.flatten()))
  accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
  # initialize figure
  plt.figure()

  # fill in label regions using scatter points
  # get (x1, x2) for plot region
  anyax = np.linspace(0.05, 0.95, num=n_axis_pts)
  xx = np.meshgrid(anyax, anyax)
  xx_vars = np.reshape(xx, (2, n_axis_pts **2))
  # get class weights from classifier model
  z = modelf(xx_vars)
  # get class label from model output
  y_hat = z.argmax(axis=1)

  for a in range(0, num_classes):
    t = np.argwhere(y_hat == a)
    t = t[:, 0]
    plt.scatter(
      xx_vars[0, t],
      xx_vars[1, t],
      s=5,
      color=accessible_color_cycle[a],
      linewidth=1.5,
      label="class:" + str(a))

  # color current class
  for a in range(0, num_classes):
    t = np.argwhere(y == a)
    t = t[:, 0]
    plt.scatter(
      x[0, t],
      x[1, t],
      s=50,
      color=accessible_color_cycle[a],
      edgecolor='k',
      linewidth=1.5,
      label="class:" + str(a))
    plt.xlabel("x1")
    plt.ylabel("x2")
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  plt.savefig("classifier_label_regions.png")
  plt.close()

lam = 10**-5
def linear_model(x,w):
	return (w[0] + jnp.dot(x.T,w[1:])).T

def multi_class_softmax(w,x,y):
   all_evals = linear_model(x,w)
   all_evals_exp = jnp.exp(all_evals)
   a = jnp.sum(all_evals_exp,axis=0)
   a = jnp.log(a)
   b = all_evals[y.astype(int).flatten(), jnp.arange(np.size(y))]
   cost = jnp.mean(a-b)
   cost = cost + lam*jnp.linalg.norm(w[1:,:],'fro')**2
   return cost

def gradient_descent_auto(g,max_its,w,x,y,diminishing_alpha=False,alpha = 1):
	# TODO: compute gradient module using jax
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w,x,y)] # container for corresponding cost function history
	#accuracy_history = [] # container for accuracy        
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

		#y_pred = jnp.sign(w[0] + jnp.dot(x.T,w[1:]))
		#count_mis = jnp.count_nonzero(y.T - y_pred)
		#accuracy = 1-(count_mis/jnp.size(y))
		#accuracy_history.append(accuracy)

	return weight_history,cost_history

def plot_cost_history(cost_history,name): 
    plt.figure() 
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='.')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.savefig(name + ".png") 
    plt.show()

def run_task3():
  # load in dataset
  data = np.loadtxt(datapath + '4class_data.csv', delimiter=',')

  # get input/output pairs
  x = data[:-1, :]
  y = data[-1:, :]

  print(np.shape(x))
  print(np.shape(y))

  show_dataset(x, y)

  # show data classified with dummy multiclass model
  def dummy_classifier_model(xs):
    y_hats = np.zeros((np.shape(xs)[1], 4))
    ys = ((1 - xs[0, :]) > xs[1, :]).astype(int)
    ys[np.where(xs[0,:] > xs[1,:])] = ys[np.where(xs[0,:] > xs[1,:])] + 2
    for i, e in enumerate(ys):
      y_hats[i,e] = 1
    return y_hats
  show_dataset_labels(x, y, dummy_classifier_model)

  # TODO: fill in your code
  w = np.random.randn(3,4)
  
  g = lambda w,x,y : multi_class_softmax(w,x,y)
  weight_history, cost_history = gradient_descent_auto(g,2000,w,x,y,False,alpha=1.0)
  plot_cost_history(cost_history, 'Task1')
  
  #Exponential normalization
  s = linear_model(x,weight_history[-1])
  s = jnp.exp(s)
  sum1 = jnp.sum(s,axis=0)
  s = s/sum1
  
  #predictions
  y_pred = jnp.argmax(s,axis=0)

  print("Final cost:")
  print(cost_history[-1])

  #misclassifications
  print('Accuracy:')
  correct = 40 - jnp.count_nonzero(y_pred-y)
  print(str((correct/jnp.size(y))*100) + '%')

  print("misclassified points:")
  print(jnp.nonzero(y_pred-y))

  def modelDummy(x):
    return linear_model(x,weight_history[-1]).T

  show_dataset_labels(x,y,modelDummy)

if __name__ == '__main__':
  run_task3()
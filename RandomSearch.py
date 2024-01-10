import numpy as np 
import matplotlib.pyplot as plt 
import os 

# set random seed to make experiment reproducible 
np.random.seed(1) 


#################### Task 1 ###################

def random_search(g, w, \
    alpha = 1, \
    max_its = 10, \
    num_samples = 1000, \
    diminishing_steplength=False):
    

    # init 
    weight_history = [w]         # container for weight history
    cost_history = [g(w)]           # container for corresponding cost function history

    # random search with max_its number of iterations 
    for k in range(1,max_its+1):  
        
        if diminishing_steplength:
            alpha = 1/float(k)
        else:
            alpha = alpha

        # TODO: put your code here. 
        random_directions = np.random.randn(num_samples,np.size(w)) 
        norm_directions = np.sqrt(np.sum(random_directions*random_directions,axis = 1))[:,np.newaxis]
        unit_directions = random_directions/norm_directions
        
        w_new = w + alpha*unit_directions
        
        g_new = np.array([g(w_element) for w_element in w_new])
        
        highest_decrease_ind = np.argmin(g_new)
        
        if g(w_new[highest_decrease_ind]) < g(w):
            descent_direction = unit_directions[highest_decrease_ind,:]
            w= w + alpha*descent_direction

        # record weights and cost evaluation
        weight_history.append(w)
        cost_history.append(g(w))

    print("Random search finished with K={K} iterations".format(K=max_its))
    return weight_history,cost_history


def run_task1():
    print("Run task 1 ....")
    # This is a test function. 
    # The random search should easily find the global optium x^*=0 
    g = lambda x: x**2
    w = np.array([-2])
    weight_history,cost_history = random_search(g, w, num_samples=5, max_its=5)
    print("weight history:")
    print(weight_history)
    
    print("cost history:")
    print(cost_history)
    
    # Uncomment the two lines below to check if your result is correct. 
    assert np.isclose(cost_history[-1], 0)==True, "The minimum cost should be zero" 
    assert np.isclose(weight_history[-1], 0)==True, "The minimum x value should be zero"
    print("Task 1 finished.")


#################### Task 2 ###################


# plot function 
def plot_cost_history(cost_history): 
    plt.figure() 
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='o')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.savefig("cost_history.png") 
    plt.show()

def con_func(w): 
    cost = None 

    g = lambda x: 100*((x[1] - x[0]**2))**2 + (x[0] - 1)**2
    cost = g(w)
    
    return cost 

def run_task2(): 
    ### Apply the random search to optimize the task2_function  
    print("Run task 2 ....") 
    
    w = np.array([-2,-2])
    g = lambda w: 100*((w[1] - w[0]**2))**2 + (w[0] - 1)**2
    
    weight_history,cost_history = random_search(g, w, num_samples=1000, max_its=50)
    plot_cost_history(cost_history)
    print("Task 2 finished.")

#################### Task 3 ###################

def compare_cost_history(costs_fixed, costs_diminished):
    plt.figure()
    plt.plot(np.arange(1, len(costs_fixed) + 1), costs_fixed, 'k-', marker="o", label="with fixed steplength")
    plt.plot(np.arange(1, len(costs_diminished) + 1), costs_diminished, "r-", marker='o', label="with diminishing steplength")
    plt.xlabel("k")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig("comparison.png")
    plt.show()


def compare_contour(weights_fixed, weights_dinimished): 
    delta = 0.001
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = 100 * (Y - X * X)**2 + (X - 1)**2 
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Contour') 

    # plot weights on top of it 
    w1 = [x[0] for x in weights_fixed]
    w2 = [x[1] for x in weights_fixed]
    ax.plot(w1, w2, 'k-', marker='o')

    w1 = [x[0] for x in weights_dinimished]
    w2 = [x[1] for x in weights_dinimished]
    ax.plot(w1, w2, 'r-', marker='o')

    fig.savefig("contour.png")


def run_task3():
    print("Run task 3 ...") 

    # TODO: fill in the code 
    # You coudl use "compare_cost_history" and "compare_contour" to 
    # produce figures used for the report.
    
    w = np.array([-2,-2])
    g = lambda w: 100*((w[1] - w[0]**2))**2 + (w[0] - 1)**2
    
    weight_history_without_diminishing,cost_history_without_diminishing = random_search(g, w, num_samples=1000, max_its=50)
    
    weight_history_with_diminishing,cost_history_with_diminishing = random_search(g, w, num_samples=1000, max_its=50, diminishing_steplength= True)
    
    compare_cost_history(cost_history_without_diminishing[10:], cost_history_with_diminishing[10:])
    
    compare_cost_history(cost_history_without_diminishing,cost_history_with_diminishing)
    
    compare_contour(weight_history_without_diminishing, weight_history_with_diminishing)
    
    final_cost_without_diminishing = con_func(weight_history_without_diminishing[len(weight_history_without_diminishing)-1])
    
    final_cost_with_diminishing = con_func(weight_history_with_diminishing[len(weight_history_with_diminishing)-1])
    
    print("without diminishing steplength final weight: " + str(weight_history_without_diminishing[len(weight_history_without_diminishing)-1]))
    
    print("diminishing steplength final weight: " + str(weight_history_with_diminishing[len(weight_history_with_diminishing)-1]))
   
    print("without diminishing steplength final cost: " + str(final_cost_without_diminishing))
    
    print("diminishing steplength final cost: " + str(final_cost_with_diminishing))
    
    print("Task 3 finished.") 






if __name__ == '__main__':
    
    run_task1() 
    run_task2()
    run_task3()

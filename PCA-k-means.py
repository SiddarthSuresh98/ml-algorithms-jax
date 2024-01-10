import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 
import math,copy
import sys

# plotting functions
import matplotlib.pyplot as plt
from matplotlib import gridspec

np.random.seed(0)

datapath = "./"

#################### Task 1 ###################

lam = 10**-5

def get_mean_centerd(x):
	x_means = np.mean(x,axis=1)[:,np.newaxis]
	x_centered = x - x_means
	return x_centered

def pca(x):
	p = float(x.shape[1])
	cov = (1/p)*(np.dot(x,x.T)) + lam * np.eye(x.shape[0])
	eig_value,eigen_vector = np.linalg.eigh(cov)
	return eig_value,eigen_vector

#pca visualizer taken from textbook (https://github.com/jermwatt/machine_learning_refined/blob/main/notes/8_Linear_unsupervised_learning/chapter_8_library) and modified according to my needs
def pca_visualizer(X,W,pcs):
    # renderer    
    fig = plt.figure(figsize = (15,5))
    
    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 2) 
    ax1 = plt.subplot(gs[0],aspect = 'equal');
    ax2 = plt.subplot(gs[1],aspect = 'equal'); 
                 
    # sphere the results
    ars = np.eye(2)
        
    # loop over panels and plot each 
    c = 1
    for ax,pt,ar in zip([ax1,ax2],[X,W],[pcs,ars]): 
        # set viewing limits for originals
        xmin = np.min(pt[0,:])
        xmax = np.max(pt[0,:])
        xgap = (xmax - xmin)*0.15
        xmin -= xgap
        xmax += xgap
        ymin = np.min(pt[1,:])
        ymax = np.max(pt[1,:])
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
    
        # scatter points
        ax.scatter(pt[0,:],pt[1,:],s = 60, c = 'k',edgecolor = 'w',linewidth = 1,zorder = 2)
   
        # plot original vectors
        vector_draw(ar[:,0].flatten(),ax,color = 'orange',zorder = 3)
        vector_draw(ar[:,1].flatten(),ax,color = 'orange',zorder = 3)

        # plot x and y axes, and clean up
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k', linewidth=1.5,zorder = 1)
        ax.axvline(x=0, color='k', linewidth=1,zorder = 1)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.grid('off')

        # set tick label fonts
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        
        # plot title
        if c == 1:
            ax.set_title('original data',fontsize = 22)
            ax.set_xlabel(r'$x_1$',fontsize = 22)
            ax.set_ylabel(r'$x_2$',fontsize = 22,rotation = 0,labelpad = 10)
        if c == 2:
            ax.set_title('encoded data',fontsize = 22)
            ax.set_xlabel(r'$c_1$',fontsize = 22)
            ax.set_ylabel(r'$c_2$',fontsize = 22,rotation = 0,labelpad = 10)
        c+=1
    plt.show()

#used in pca visualizer taken from textbook
def vector_draw(vec,ax,**kwargs):
    color = 'k'
    if 'color' in kwargs:
        color = kwargs['color']
    zorder = 3 
    if 'zorder' in kwargs:
        zorder = kwargs['zorder']
        
    veclen = math.sqrt(vec[0]**2 + vec[1]**2)
    head_length = 0.25
    head_width = 0.25
    vec_orig = copy.deepcopy(vec)
    vec = (veclen - head_length)/veclen*vec
    ax.arrow(0, 0, vec[0],vec[1], head_width=head_width, head_length=head_length, fc=color, ec=color,linewidth=3,zorder = zorder)
      

def run_task1(): 
	# load in dataset
	csvname = datapath + '2d_span_data.csv'
	x = np.loadtxt(csvname, delimiter = ',')

	print(np.shape(x)) # (2, 50)

    # TODO: fill in your code 
	x_mean_centered = get_mean_centerd(x)

	eig_value,eigen_vector = pca(x_mean_centered)

	print("eigen values:")
	print(eig_value)

	print("eigen_vectors:")
	print(eigen_vector)

	x_pca = np.dot(eigen_vector.T, x_mean_centered)
     
	pca_visualizer(x_mean_centered,x_pca,eigen_vector)


#################### Task 2 ###################

#referenced from textbook (https://github.com/jermwatt/machine_learning_refined/blob/main/notes/8_Linear_unsupervised_learning/)
def update_assignments(data,centroids):
    P = np.shape(data)[1]
    assignments = []
    for p in range(P):
        x_p = data[:,p][:,np.newaxis]
        diffs = np.sum((x_p - centroids)**2,axis = 0)
        ind = np.argmin(diffs)
        assignments.append(ind)
    return np.array(assignments)
       
# update centroid locations, referenced from textbook(https://github.com/jermwatt/machine_learning_refined/blob/main/notes/8_Linear_unsupervised_learning/)
def update_centroids(data,old_centroids,assignments):
    K = old_centroids.shape[1]
    centroids = []
    for k in range(K):
        S_k = np.argwhere(assignments == k)
        c_k = 0
        if np.size(S_k) > 0:
            c_k = np.mean(data[:,S_k],axis = 1)
        else:
            c_k = copy.deepcopy(old_centroids[:,k])[:,np.newaxis]
        centroids.append(c_k)
    centroids = np.array(centroids)[:,:,0]
    return centroids.T

# main k-means function, referenced from textbook(https://github.com/jermwatt/machine_learning_refined/blob/main/notes/8_Linear_unsupervised_learning/chapter_8_library/)
def my_kmeans(data,centroids,max_its):   
    for j in range(max_its):
        assignments = update_assignments(data,centroids)
        centroids = update_centroids(data,centroids,assignments)
    assignments = update_assignments(data,centroids)
    return centroids,assignments

def plotKmeans(data,centroids,assignments):
    K = len(centroids.T)
    colors =  [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.75, 0.75, 0.75],'mediumaquamarine']
    plt.title("Clustered Data after K Means Clustering")
    for k in range(K):
        ind = np.argwhere(assignments == k)
        if np.size(ind) > 0:
            ind = [s[0] for s in ind]    
            plt.scatter(data[0,ind],data[1,ind],color = colors[k],s = 100,edgecolor = 'k',linewidth = 1,zorder = 2, label=f'Cluster {k+1}') 
    for k in range(K):
        plt.scatter(centroids[0,k],centroids[1,k],c = colors[k],s = 400,edgecolor ='k',linewidth = 2,marker=(5, 1),zorder = 3, label=f'Centroid {k+1}')
    plt.legend()
    plt.show()

#referenced from textbook(https://github.com/jermwatt/machine_learning_refined/blob/main/notes/8_Linear_unsupervised_learning/chapter_8_library/)
def compute_average(data,centroids,assignments):
    P = len(assignments)
    K = np.shape(centroids)[1]
    error = 0
    for k in range(K):
        centroid = centroids[:,k]
        ind = np.argwhere(assignments == k)
        if np.size(ind) > 0:
            ind = [s[0] for s in ind]    
            for i in ind:
                pt = data[:,i]
                error += np.linalg.norm(centroid - pt)
    error /= float(P)
    return error

#referenced from textbook and modified according to my needs(https://github.com/jermwatt/machine_learning_refined/blob/main/notes/8_Linear_unsupervised_learning/chapter_8_library/)
def scree_plot(data,K_range,max_its):
    colors =  [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.75, 0.75, 0.75],'mediumaquamarine']
    fig = plt.figure(figsize = (8,3))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0]); 

    K_errors = []
    for k in K_range:
        errors = []
        for j in range(5):
            P = np.shape(data)[1]
            random_inds = np.random.permutation(P)[:k]
            init_centroids = data[:,random_inds]
            all_centroids,all_assignments = my_kmeans(data,init_centroids,max_its-1)
            centroids = all_centroids
            assignments = all_assignments
            error = compute_average(data,centroids,assignments)
            errors.append(error)
        best_ind = np.argmin(errors)
        K_errors.append(errors[best_ind]) 
    ax.plot(K_range,K_errors,'ko-')
    ax.set_title('cost value')
    ax.set_xlabel('number of clusters')
    ax.set_xticks(K_range)
    plt.show()

#returns distance between two points
def distance_points(p1,p2):
    return np.linalg.norm(p1-p2)

#gets k cluster centroids from the initial centroid provided. Each centroid is ensured to be at a maximum distance from previous centroids. 
def initializeCentroids(data,initial_centroid,k):
    centroids = list(initial_centroid.T)
    for i in range (k-1):
        distance =[]
        for j in range(np.shape(data)[1]):
            p = data[:,j]
            d = sys.maxsize
            for k in range(len(centroids)):
                distance_from_centroid = distance_points(p,centroids[k])
                d = min(d,distance_from_centroid)
            distance.append(d)
        dist = np.array(distance)
        new_centroid = data[:,np.argmax(dist)]
        centroids.append(new_centroid)
        distance = []
    centroids = np.array(centroids)
    return centroids

def run_task2(): 
	# Loading the data
    P = 50 # Number of data points
    blobs = datasets.make_blobs(n_samples = P, centers = 3, random_state = 10)
    data = np.transpose(blobs[0])

    # TODO: fill in your code
    #plot the initial data
    plt.title("Data")
    plt.scatter(data[0,:], data[1,:])
    plt.show()

    r_c = np.arange(np.shape(data)[1])
    pt = np.random.permutation(r_c)[:1]
    #initial centroid is a random data point in the dataset
    initial_centroid = data[:,pt]
    
    #Initializing k-1 centroids ensuring they are as far apart as possible to form better clusters.
    initial_centroids = initializeCentroids(data,initial_centroid,3)
    initial_centroids = initial_centroids.T
    
    #perform kmeans clustering.
    centroids,assignments = my_kmeans(data,initial_centroids,100)

    #color plot of clusters
    plotKmeans(data,centroids,assignments)
    
    #Scree plot to find optimal number of clusters
    K_range = [1,2,3,4,5,6,7,8,9,10]
    scree_plot(data,K_range,max_its=100)

if __name__ == '__main__':
	run_task1()
	run_task2() 
import numpy as np
import random

BEST_K_CLASSIC_KMEANS = 85

class KMeans(object):
    """
    K-Means clustering class.
    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K=BEST_K_CLASSIC_KMEANS, max_iters=100, distance = "euclidian", kmeansPlus = False):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        if(distance!="euclidian" and distance!="manhattan"):
            raise ValueError("distance should be euclidian or manhattan")
        self.K = K
        self.max_iters = max_iters
        self.centers = None
        self.cluster_center_labels = None
        self.distance = distance
        self.kmeansPlus = kmeansPlus

            
    def init_centers(self, data, K):
        """
        Randomly pick K data points from the data as initial cluster centers.
        Arguments: 
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        #normal (random) initialisation
        if(not self.kmeansPlus):
            # Select the first K random index
            random_idx = np.random.permutation(data.shape[0])[:K]
            # Use these index to select centers from data
            centers = data[random_idx[:K]]
            return centers
        #kmeans++:
        else:
            N = data.shape[0]
            #first center is random
            centers = np.zeros((K, data.shape[1]))
            centers[0]=data[random.randint(0, N)]
            #compute other centers such:
            for k in range (1,K):
                #compute distances of data points to already computed centers
                distances = self.compute_distance(data, centers, True, k)
                #take distance of the closest center
                closest_center_distance = np.amin(distances, axis=1)
                #choose a data point to be next center, such that the chance a data point is chosen 
                # is proportional to the square distance of the closest center for that point
                centers[k]= data[random.choices(np.arange(N), weights=closest_center_distance**2)[0]]
            return centers
    
    def compute_distance(self, data, centers, specialK=False, givenK=0):
        """
        Compute the euclidean distance between each datapoint and each center.
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
            specialK : boolean to say if the givenK argument should be taken into account or not as the actual number of centers
            givenK : positive integer, indicates the actual number of centers that are taken into account 
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        K = centers.shape[0]
        #redefine K in cases number of centers to take into account was given in argument
        if(specialK):
            K = givenK
        # Here, we will loop over the cluster
        distances = np.zeros((N, K))
        for k in range(K):
            # Compute the distance for each data to each center
            center = centers[k]
            if(self.distance == "euclidian"):
                distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))
            elif (self.distance == "manhattan"):
                distances[:, k] = np.abs(data - center).sum(axis=1)
            
        return distances
    
    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def compute_centers(self, data, cluster_assignments, K, old_centers):
        """
        Compute the center of each cluster based on the assigned points.
        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
            old_centers : centers at previous iteration
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = np.zeros((K, data.shape[1]))
        for i in range(K):
            # Compute the mask for the cluster
            mask=cluster_assignments==i
            # Compute the number of points associated to the cluster
            length = np.sum(mask)
            #if no point is associated to the cluster, keep the old center
            if(length==0):
                centers[i]=old_centers[i]
            #else compute the new center as center of mass of the points in the cluster
            else:
                centers[i] = (np.sum(data[mask],axis=0)/length).reshape(1,-1)
        return centers



    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """   
         # Initialize the centers
        centers = self.init_centers(data, self.K)
        # Initialize the distances 
        distances=np.zeros((data.shape[0], self.K))
        # Initialize the cluster assignments
        cluster_assignments=np.zeros(data.shape[0])
        # Loop over the iterations
        for i in range(max_iter):
            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}...")
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration

            distances = self.compute_distance(data, old_centers)
            cluster_assignments = self.find_closest_cluster(distances)
            #print(data.shape, closest_cluster.shape)
            centers = self.compute_centers(data, cluster_assignments, self.K, old_centers)
        
            # End of the algorithm if the centers have not moved (hint: use old_centers and look into np.all)
            if np.all(centers==old_centers):  
                print(f"K-Means has converged after {i+1} iterations!")
                break
            
        return centers, cluster_assignments
    
    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.
        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        cluster_center_label = np.zeros(centers.shape[0])
        #for each center, find the most frequent label in the cluster:
        for i in range(len(centers)):
            #take true labels of points in cluster i
            true_labels_mask = true_labels[cluster_assignments == i]
            if(true_labels.shape[0]!=0):
                bincount = np.bincount(true_labels_mask)
                label=-1 #default label
                if(bincount.shape[0]>0):
                    #compute most frequent label
                    label = np.argmax(np.bincount(true_labels_mask))
                cluster_center_label[i] = label
            else:
                #default label is -1
                cluster_center_label[i] = -1
        return cluster_center_label

    def predict_with_centers(self, cluster_center_label, cluster_assignments):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.
        Arguments: 
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns: 
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        # Convert cluster index to label
        new_labels = cluster_center_label[cluster_assignments]
        return new_labels
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.
        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        #compute centers and cluster assignments
        self.centers, cluster_assignments = self.k_means(training_data)
        #compute cluster centers labels
        self.cluster_center_label=self.assign_labels_to_centers(self.centers, cluster_assignments, training_labels)
        #predict labels
        pred_labels=self.predict_with_centers(self.cluster_center_label, cluster_assignments)
        return pred_labels
        
    def average_within_cluster_sum_of_squares(self, data):
        """
        This method will be used to choose the right k in kmeans_fit_k when using the elbow method. It just returns the current 
        average within-cluster sum of squared distances

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            averahe within cluster sum of squares
        """
        #compute distances
        distances = self.compute_distance(data, self.centers) ** 2
        #find closest cluster for each data point
        cluster_assignments = self.find_closest_cluster(distances)
        val = 0
        #for each cluster, compute the sum of squared distances
        for i in range(self.K):
            val += np.sum(distances[cluster_assignments == i, i] ** 2)
        #return average
        return val / self.K

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.
        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        #find closest cluster for each test data point
        cluster_assignments = self.find_closest_cluster(self.compute_distance(test_data, self.centers))
        #predict label of each test data point
        pred_labels = self.predict_with_centers(self.cluster_center_label, cluster_assignments)
        return pred_labels
    

from src.methods.kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np

def plot_all_possibilities(xtrain,xtest, ytrain,ytest, number_of_k, accuracy_fn):
        """
        plot the accuracies for the 4 different possibilities of the kmeans class.
        Arguments: 
            xtrain: the training data, shape is (N,D) where N is the number of data points, D the number of features
            xtest: the test data, shape is (N',D') where N is the number of data points, D the number of features
            ytrain: the training labels, shape is (N,)
            ytest: the test labels, shape is (N',)  
            number_of_k: the number of k values to test
            accuracy_fn: the function to compute the accuracy
        """
        #initialize arrays that will contain accuracies results
        accuracies_classic_manhattan = [0] * number_of_k
        accuracies_plus_manhattan = [0] * number_of_k
        accuracies_classic_euclidian = [0] * number_of_k
        accuracies_plus_euclidian = [0] * number_of_k
        print("Computing data points")
        k_values = [k for k in range(1, number_of_k+1)]
        #loop over all k values
        for k in k_values:
            print("Testing with k = ", k)
            #manhattan:
            #classic k means
            km_classic = KMeans(k, distance="manhattan", kmeansPlus=False)
            #fit
            km_classic.fit(xtrain, ytrain)
            #predict
            pred=km_classic.predict(xtest)
            #compute accuracy
            accu = accuracy_fn(pred, ytest)
            print("accu for classic :", accu)
            #store accuracy
            accuracies_classic_manhattan[k-1]=accu

            #kmeans ++ 
            km_plus = KMeans(k, distance="manhattan", kmeansPlus=True)
            #fit
            km_plus.fit(xtrain, ytrain)
            #predict
            pred=km_plus.predict(xtest)
            #compute accuracy
            accu = accuracy_fn(pred, ytest)
            print("accu for kmeans++ :", accu)
            #store accuracy
            accuracies_plus_manhattan[k-1]=accu
            

            #euclidian : 
            #classic k means
            km_classic = KMeans(k, kmeansPlus=False)
            #fit
            km_classic.fit(xtrain, ytrain)
            #predict
            pred=km_classic.predict(xtest)
            #compute accuracy
            accu = accuracy_fn(pred, ytest)
            print("accu for classic :", accu)
            #store accuracy
            accuracies_classic_euclidian[k-1]=accu
            
            #kmeans ++ 
            km_plus = KMeans(k, kmeansPlus=True)
            #fit
            km_plus.fit(xtrain, ytrain)
            #predict
            pred=km_plus.predict(xtest)
            #compute accuracy
            accu = accuracy_fn(pred, ytest)
            print("accu for kmeans++ :", accu)
            #store accuracy
            accuracies_plus_euclidian[k-1]=accu
            

        #plot accuracies
        plt.plot(k_values, accuracies_classic_manhattan, marker="*", color="b",linestyle='dashed', label="classic kmeans manhattan")
        plt.plot(k_values, accuracies_plus_manhattan, marker="*", color="b", label="kmeans ++ manhattan")
        plt.plot(k_values, accuracies_classic_euclidian, marker="*", color="r",linestyle='dashed', label="classic kmeans euclidian")
        plt.plot(k_values, accuracies_plus_euclidian, marker="*", color="r", label="kmeans ++ euclidian")
        plt.legend()
        plt.ylabel('accuray on validation data')
        plt.xlabel('Values of K')
        plt.title('cross validation')
        plt.show()
        plt.close()

def elbow_method(xtrain, ytrain, maxK):
    """
        plot the average within cluster sum of squares for different k on training data for classical kmeans with euclidean distance (used to find the best k according to the elbow method)
        Arguments: 
            xtrain: the training data, shape is (N,D) where N is the number of data points, D the number of features
            ytrain: the training labels, shape is (N,)
            maxK: the maximum number of clusters to test
    """
    #initialize arrays that will contain average within cluster sum of squares results
    average_within_cluster_distances = [0] * maxK
    k_values = [k for k in range(1, maxK+1)]
    #compute average within cluster sum of squares for each k
    for k in k_values:
        print("Testing with k = ", k)
        km_classic = KMeans(k, distance="euclidian", kmeansPlus=False)
        #fit
        km_classic.fit(xtrain, ytrain)
        #compute average within cluster sum of squares
        average_within_cluster_distance = km_classic.average_within_cluster_sum_of_squares(xtrain)
        #store average within cluster sum of squares
        average_within_cluster_distances[k-1]=average_within_cluster_distance
    #plot results
    plt.plot(k_values, average_within_cluster_distances, marker="*", color="r",linestyle='dashed')
    plt.ylabel('average within cluster sum of squares')
    plt.xlabel('Values of K')
    plt.title('elbow method')
    plt.show()
    plt.close()

    
    
def KFold_cross_validation(xtrain, ytrain, number_of_folds, k, accuracy_fn, normalize_fn):
    '''
    K-Fold Cross validation function for K-NN

    Inputs:
        xtrain : training data, shape (NxD)
        ytrain: training labels, shape (N,)
        number_of_folds: number of folds (K in K-fold)
        k: number of clusters (the hyperparameter)
        accuracy_fn: the function to compute the accuracy
        normalize_fn: the function to normalize the data

    Returns:
        Average validation accuracy for the selected k.
    '''
    N = xtrain.shape[0]
    
    accuracies = []  # list of accuracies
    for fold_ind in range(number_of_folds):
        #Split the data into training and validation folds:
        
        #all the indices of the training dataset
        all_ind = np.arange(N)
        split_size = N // number_of_folds
        
        # Indices of the validation and training examples
        val_ind = all_ind[fold_ind * split_size : (fold_ind + 1) * split_size]
        train_ind = np.setdiff1d(all_ind, val_ind)
        #training set for this fold
        X_train_fold = xtrain[train_ind, :]
        Y_train_fold = ytrain[train_ind]
        #validation set for this fold
        X_val_fold = xtrain[val_ind, :]
        Y_val_fold = ytrain[val_ind]
        #normalize everything
        mean_val = X_train_fold.mean(axis=0,keepdims=True)
        std_val  = X_train_fold.std(axis=0,keepdims=True)
        norm_train_data = normalize_fn(X_train_fold, mean_val, std_val)
        norm_test_data  = normalize_fn(X_val_fold, mean_val, std_val)
        #train
        km_classic = KMeans(k, kmeansPlus=False)
        #fit
        km_classic.fit(norm_train_data, Y_train_fold)
        #predict 
        Y_val_fold_pred=km_classic.predict(norm_test_data)
        #compute accuracy
        acc = accuracy_fn(Y_val_fold_pred, Y_val_fold)
        accuracies.append(acc)
    
    #Find the average validation accuracy over K:
    ave_acc = np.mean(np.array(accuracies))
    return ave_acc   


def run_cv_for_hyperparam(xtrain, ytrain, number_of_folds, accuracy_fn, normalize_fn):
    '''
    K-Fold Cross validation function for K-NN

    Inputs:
        xtrain : training data, shape (NxD)
        ytrain: training labels, shape (N,)
        number_of_folds: number of folds (K in K-fold)
        accuracy_fn: the function to compute the accuracy
        normalize_fn: the function to normalize the data
    Returns:
        model_performance: a list of validation accuracies corresponding to the k-values     
    '''
    #list of k to test
    k_list = range(1, 100, 4)          
    model_performance = [] 
    #for each k, compute the average validation accuracy
    for k in k_list:
        print("Testing with k = ", k)
        model_performance.append(KFold_cross_validation(xtrain,ytrain,number_of_folds,k, accuracy_fn, normalize_fn))
    #plot the results
    plt.plot(k_list, model_performance)
    plt.xlabel("k value $k$")
    plt.xticks(k_list)
    plt.ylabel("Performance (accuracy)")
    plt.title("Performance on the validation set for different values of $k$")
    plt.show()
    plt.close()

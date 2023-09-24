import argparse
import math
import numpy as np 
import matplotlib.pyplot as plt
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
import kmeans_plots 
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn

"""
File constants
"""
OPT_METHOD = "svm"  # Details in report


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    xtrain = normalize_fn(xtrain, np.mean(xtrain), np.std(xtrain))
    xtest = normalize_fn(xtest, np.mean(xtest), np.std(xtest))

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        xtrain_temp = xtrain[: math.floor(xtrain.shape[0]*0.8)]
        ytrain_temp = ytrain[: math.floor(xtrain.shape[0]*0.8)]
        xtest = xtrain[math.floor(xtrain.shape[0]*0.8):]
        ytest = ytrain[math.floor(xtrain.shape[0]*0.8):]
        xtrain = xtrain_temp
        ytrain = ytrain_temp
        
        pass

    #append bias term
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)
    
    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":  ### WRITE YOUR CODE HERE

        #kmeans for the best k we found
        km_classic = KMeans(distance="euclidian", kmeansPlus=False)
        method_obj = km_classic

        #ADDITIONAL POSSIBILITIES TO RUN:  
        
        #if you want the plot for the elbow method, run the following code:
        #kmeans_plots.elbow_method(xtrain, ytrain, 20)

        #if you want to plot the accuracy on validation set for the 4 possibilities of k means, run the following code:
        #kmeans_plots.plot_all_possibilities(xtrain, xtest, ytrain, ytest, 35, accuracy_fn)

        #if you want the plot of accuracies using Kfold cross validation for different number of cluster values:, run the following code :  
        #kmeans_plots.run_cv_for_hyperparam(xtrain, ytrain, 5, accuracy_fn, normalize_fn)
    
        pass
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression()

    elif args.method == "svm":  
        method_obj = SVM(args.svm_c, args.svm_kernel, args.svm_gamma, args.svm_degree, args.svm_coef0)

    else:
        pass


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
        
    # Predict on unseen data
    preds = method_obj.predict(xtest)


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    if args.visualize and args.method == 'logistic_regression':
        minLr = args.minLr
        maxLr = args.maxLr
        minIter = args.minIter
        maxIter = args.maxIter
        steps = args.steps
        assert minLr <= maxLr, "Minimum learning rate superior to maximum learning rate, exiting"
        assert minIter <= maxIter, "Minimum maximum number of iterations superior to maximum maximum number of iterations, exiting"
        assert steps > 0, "Number of steps not strictly positive, exiting"
        method_obj.visualize(training_data=xtrain, training_labels=ytrain, test_data=xtest, test_labels=ytest,
                             minLr=minLr, maxLr=maxLr, minIter=minIter, maxIter=maxIter, steps=steps)


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    # For all the svm arguments the default values are set to None, and the real default value are the computed optimal parameters, handled in svm.py
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default=OPT_METHOD, type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=None, help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default=None, help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=None, help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=None, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=None, help="coef0 in polynomial SVM method")

    # Feel free to add more arguments here if you need!
    # Visualize arguments
    parser.add_argument('--visualize', type=bool, default=False, help="logistic regression : "
                                                                      "whether to plot accuracy in function of the "
                                                                      "learning rate and the maximum number of iter")
    parser.add_argument('--minLr', type=float, default=1e-5, help="logistic regression, minimum learning rate")
    parser.add_argument('--maxLr', type=float, default=5e-3, help="logistic regression, maximum learning rate, >minLr")
    parser.add_argument('--minIter', type=int, default=50,
                        help="logistic regression, minimum maximum number of iterations")
    parser.add_argument('--maxIter', type=int, default=500,
                        help="logistic regression, maximum maximum number of iterations, >minIter")
    parser.add_argument('--steps', type=int, default=10, help="number of steps to go from minLr to maxLr, nb of steps"
                                                               "to go from minIter to maxIter, >0")

    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

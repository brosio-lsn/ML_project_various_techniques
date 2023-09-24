import argparse
import math
import numpy as np
from src.plots import plot_all_possibilities
from src.drawings import test
from torchinfo import summary
import matplotlib.pyplot as plt
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

# Best parameters found using cross-validation, see report for more details
LR_MLP = 1e-1
MAX_ITERS_MLP = 30

LR_CNN = 0.125
MAX_ITERS_CNN = 30

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
    #  normalize, add bias, etc. (no need to append bias here)

    # Make a validation set
    if not args.test:
        xtrain_temp = xtrain[: math.floor(xtrain.shape[0]*0.8)]
        ytrain_temp = ytrain[: math.floor(xtrain.shape[0]*0.8)]
        xtest = xtrain[math.floor(xtrain.shape[0]*0.8):]
        ytest = ytrain[math.floor(xtrain.shape[0]*0.8):]
        xtrain = xtrain_temp
        ytrain = ytrain_temp
        pass

    mean = np.mean(xtrain)
    std = np.std(xtrain)
    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean, std)
    
    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        exvar = pca_obj.find_principal_components(xtrain)
        print(f'The total variance explained by the first {args.pca_d} principal components is {exvar:.3f} %')
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)



    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)

        if args.nn_type == "mlp":
            model = MLP(xtrain.shape[1], n_classes, [256, 128, 64])

            # Setting the params to optimized for MLP if not specified
            args.lr = args.lr if args.lr != None else LR_MLP
            args.max_iters = args.max_iters if args.max_iters != None else MAX_ITERS_MLP
            #IF YOU WANT TO PLOT ACCURACY VS LEARNING RATE FOR MLP
            #plot_all_possibilities(xtrain, xtest, ytrain, ytest, np.array([1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 0.5, 1, 5]), accuracy_fn, args.max_iters, args.nn_batch_size)
        elif args.nn_type == "cnn":
            
            # Reshape the data for CNN (making square images)
            side = int(np.sqrt(xtrain.shape[1]))
            xtrain = xtrain.reshape(xtrain.shape[0], 1, side, side)
            xtest = xtest.reshape(xtest.shape[0], 1, side, side)
            
            model = CNN(1, n_classes, side=side)

            # Setting the params to optimized for MLP if not specified
            args.lr = args.lr if args.lr != None else LR_CNN
            args.max_iters = args.max_iters if args.max_iters != None else MAX_ITERS_CNN
        
        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    elif args.method == "svm":  
        method_obj = SVM(args.svm_c, args.svm_kernel, args.svm_gamma, args.svm_degree, args.svm_coef0)





    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    if args.draw: # If you want to try drawing a symbol
        #USE WITH MLP, WITHOUT PCA
        test(method_obj, mean, std, normalize_fn)


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


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="../dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="nn", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=40, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=None, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=None, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=140, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--draw', action="store_true", help="draw a symbol from the dataset")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

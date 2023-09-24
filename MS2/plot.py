import argparse
import math
import numpy as np
from src.plots import plot_all_possibilities
# from src.drawings import test
from torchinfo import summary

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import matplotlib.pyplot as plt

def main(args):
    lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    accs = []
        
        
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)

    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    # print("SHAPE",xtrain.shape)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc. (no need to append bias here)

    # Make a validation set

    xtrain_temp = xtrain[: math.floor(xtrain.shape[0]*0.8)]
    ytrain_temp = ytrain[: math.floor(xtrain.shape[0]*0.8)]
    xtest = xtrain[math.floor(xtrain.shape[0]*0.8):]
    ytest = ytrain[math.floor(xtrain.shape[0]*0.8):]
    xtrain = xtrain_temp
    ytrain = ytrain_temp

    
    mean = np.mean(xtrain)
    std = np.std(xtrain)
    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean, std)

    n_classes = get_n_classes(ytrain)


    model = MLP(xtrain.shape[1], n_classes, [256, 128, 64])  
    
    summary(model)

    for lr in lrs:
        # Trainer object
        method_obj = Trainer(model, lr=lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


        ## 4. Train and evaluate the method

        # Fit (:=train) the method on the training data
        preds_train = method_obj.fit(xtrain, ytrain)
        # if you want to test your own digits : 
        # test(method_obj, mean, std, normalize_fn)
            
        # Predict on unseen data
        preds = method_obj.predict(xtest)


        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        accs.append(acc)

    plt.plot(lrs, accs)
    plt.xlabel("Learning Rate")
    plt.xscale("log")
    plt.ylabel("Accuracy")
    plt.show()
    
    
if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="../dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--max_iters', type=int, default=30, help="max iters for methods which are iterative")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    
    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

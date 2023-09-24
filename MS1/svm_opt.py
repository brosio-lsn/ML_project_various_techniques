import argparse
import math
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.data import load_data
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn

C_RBF, GAMMA_RBF = 3, 0.00075
C_POLY, GAMMA_POLY, DEGREE_POLY, COEF0_POLY = 45, 1, 2, 750
C_LIN = 0.006 #94.159% val

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
    xtrain = normalize_fn(xtrain, np.mean(xtrain), np.std(xtrain))
    xtest = normalize_fn(xtest, np.mean(xtest), np.std(xtest))
    
    # Splitting the training set in training and validation for cross validation (80-20 ratio)
    
    ytrain80 = ytrain[: math.floor(xtrain.shape[0]*0.8)]
    xtrain80 = xtrain[: math.floor(xtrain.shape[0]*0.8)]
    xvalidation20 = xtrain[math.floor(xtrain.shape[0]*0.8):]
    yvalidation20 = ytrain[math.floor(xtrain.shape[0]*0.8):]
    
    


    if args.goal == "none" :
        method_obj = SVM(args.c, args.kernel, args.gamma, args.degree, args.coef0)
        preds_train = method_obj.fit(xtrain80, ytrain80)

        preds = method_obj.predict(xvalidation20)

        acc = accuracy_fn(preds_train, ytrain80)
        macrof1 = macrof1_fn(preds_train, ytrain80)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, yvalidation20)
        macrof1 = macrof1_fn(preds, yvalidation20)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.goal == "c_lin" : ## 0.006, 94.159
        NUMBER_OF_SAMPLES = 10
        #c_values = np.logspace(-3, -1, NUMBER_OF_SAMPLES)
        c_values = np.linspace(0.002, 0.008, NUMBER_OF_SAMPLES)
        accuracies = np.zeros(NUMBER_OF_SAMPLES)
        
        for i in range(NUMBER_OF_SAMPLES):
            print("Testing with C = ", c_values[i])
            method_obj = SVM(C=c_values[i], kernel="linear")
            preds_train = method_obj.fit(xtrain80, ytrain80)

            preds = method_obj.predict(xvalidation20)

            acc = accuracy_fn(preds_train, ytrain80)
            macrof1 = macrof1_fn(preds_train, ytrain80)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalidation20)
            macrof1 = macrof1_fn(preds, yvalidation20)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
            accuracies[i] = acc
        
        fig, graph = plt.subplots()
        graph.plot(c_values, accuracies)
        graph.set_xscale('log')
        graph.set_xlabel('Values of C')
        graph.set_ylabel('Accuracy [%]')
        graph.set_title('Accuracy with different C values')
        plt.show()

    elif args.goal == "gamma_c_rbf" :
        NUMBER_OF_SAMPLES = 5
        #c_values = np.logspace(-2, 10, NUMBER_OF_SAMPLES)
        c_values = np.linspace(5, 15, NUMBER_OF_SAMPLES)
        #gamma_values = np.logspace(-9, 3, NUMBER_OF_SAMPLES)
        gamma_values = np.linspace(0.0005, 0.0015, NUMBER_OF_SAMPLES)
        #c_values[0] = C_RBF
        #gamma_values[0] = GAMMA_RBF
        
        accuracies = np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_SAMPLES))
        
        for c_index in range(NUMBER_OF_SAMPLES):
            for gamma_index in range(NUMBER_OF_SAMPLES):
                
                print("\nTesting with C = ", c_values[c_index], " and gamma = ", gamma_values[gamma_index])
                method_obj = SVM(c_values[c_index], "rbf", gamma_values[gamma_index])
                preds_train = method_obj.fit(xtrain80, ytrain80)

                preds = method_obj.predict(xvalidation20)

                acc = accuracy_fn(preds_train, ytrain80)
                macrof1 = macrof1_fn(preds_train, ytrain80)
                print(f"Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

                acc = accuracy_fn(preds, yvalidation20)
                macrof1 = macrof1_fn(preds, yvalidation20)
                print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                
                accuracies[c_index, gamma_index] = acc
        
        plt.imshow(accuracies, extent=[c_values[0], c_values[-1], gamma_values[0], gamma_values[-1]],
           origin='lower', cmap='viridis', aspect='auto', vmin=np.min(accuracies), vmax=np.max(accuracies))

        plt.xlabel('Values of gamma')
        plt.ylabel('Values of C')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Accuracy with different C and gamma')
        plt.colorbar()
        plt.show()
    
    elif args.goal == "c_rbf" : 
        NUMBER_OF_SAMPLES = 10
        c_values = np.logspace(0, 2, NUMBER_OF_SAMPLES)
        accuracies = np.zeros((NUMBER_OF_SAMPLES))
        for c_index in range(NUMBER_OF_SAMPLES):
            print("\nTesting with C = ", c_values[c_index])
            method_obj = SVM(c_values[c_index], "rbf", GAMMA_RBF)
            
            preds_train = method_obj.fit(xtrain80, ytrain80)

            preds = method_obj.predict(xvalidation20)

            acc = accuracy_fn(preds_train, ytrain80)
            macrof1 = macrof1_fn(preds_train, ytrain80)
            print(f"Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalidation20)
            macrof1 = macrof1_fn(preds, yvalidation20)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
            
            accuracies[c_index] = acc
            
        fig, graph = plt.subplots()
        graph.plot(c_values, accuracies)
        graph.set_xscale('log')
        graph.set_xlabel('Values of C')
        graph.set_ylabel('Accuracy [%]')
        graph.set_title('Accuracy with different C values and gamma = 0.0025')
        plt.show()
        
    elif args.goal == "poly":
        '''
        Does a gridsearch for the best C - gamma combo, then run this on degrees 2 to 10 and varying coef0
        '''
    
        NUMBER_OF_SAMPLES = 10
        values = range(2, 10)
        accuracies = np.zeros((NUMBER_OF_SAMPLES))
        for index in range(NUMBER_OF_SAMPLES):
            print("\nTesting with degree = ", values[index])
            method_obj = SVM(C_POLY, "poly", GAMMA_POLY, values[index], COEF0_POLY)
            
            preds_train = method_obj.fit(xtrain80, ytrain80)

            preds = method_obj.predict(xvalidation20)

            acc = accuracy_fn(preds_train, ytrain80)
            macrof1 = macrof1_fn(preds_train, ytrain80)
            print(f"Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalidation20)
            macrof1 = macrof1_fn(preds, yvalidation20)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
            
            accuracies[index] = acc
        print(acc)
        
        
        '''
        degree = np.array([2, 3, 4, 5, 6, 7, 8, 9])
        accuracies = np.zeros(NUMBER_OF_SAMPLES)
        
        for i in range(NUMBER_OF_SAMPLES):
            print("Testing with degree = ", degree[i])
            method_obj = SVM(C_POLY, "poly", GAMMA_POLY, degree[i], 0)
            preds_train = method_obj.fit(xtrain80, ytrain80)

            preds = method_obj.predict(xvalidation20)

            acc = accuracy_fn(preds_train, ytrain80)
            macrof1 = macrof1_fn(preds_train, ytrain80)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, yvalidation20)
            macrof1 = macrof1_fn(preds, yvalidation20)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
            accuracies[i] = acc
        
        fig, graph = plt.subplots()
        graph.plot(degree, accuracies)
        graph.set_xlabel('Degree')
        graph.set_ylabel('Accuracy [%]')
        graph.set_title('Accuracy with different degree polynomials')
        plt.show()
    '''    
        
    elif args.goal == "test_poly_opt":
        method_obj = SVM(C_POLY, "poly", GAMMA_POLY, DEGREE_POLY, COEF0_POLY)
        preds_train = method_obj.fit(xtrain80, ytrain80)

        val_preds = method_obj.predict(xvalidation20)
        test_preds = method_obj.predict(xtest)

        acc = accuracy_fn(preds_train, ytrain80)
        macrof1 = macrof1_fn(preds_train, ytrain80)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(val_preds, yvalidation20)
        macrof1 = macrof1_fn(val_preds, yvalidation20)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        
        acc = accuracy_fn(test_preds, ytest)
        macrof1 = macrof1_fn(test_preds, ytest)
        print(f"Testing set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    
    else:
        return 
    

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', default="none", type=str, help="The thing you want to plot, can be 'none', 'c_lin', 'gamma_c_rbf', 'poly'")
    parser.add_argument('--data', default="./dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

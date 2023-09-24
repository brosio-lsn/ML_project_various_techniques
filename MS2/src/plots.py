from src.methods.deep_network import MLP, Trainer
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

def plot_all_possibilities(xtrain,xtest, ytrain,ytest, learning_rates, accuracy_fn, epochs, batech_size):
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
        n_classes = get_n_classes(ytrain)
        accuracies = [0] * learning_rates.shape[0]
        #loop over all k values
        for it, lr in enumerate(learning_rates):
            print("Testing with lr = ", lr)
            #manhattan:
            #classic k means
            model = MLP(xtrain.shape[1], n_classes, [512, 256, 128])
            method_obj = Trainer(model, lr=lr, epochs=epochs, batch_size=batech_size)
            #fit
            method_obj.fit(xtrain, ytrain)
            #predict
            pred=method_obj.predict(xtest)
            #compute accuracy
            accu = accuracy_fn(pred, ytest)
            print("accu for lr", lr, "is", accu)
            #store accuracy
            accuracies[it]=accu

        #plot accuracies
        print(learning_rates)
        print(accuracies)

        plt.plot(learning_rates, accuracies, marker="*", color="b",linestyle='dashed')
        plt.legend()
        matplotlib.rcParams.update({'font.size': 18})
        plt.semilogx(learning_rates, accuracies, marker="*", color="b",linestyle='dashed')
        plt.ylabel('accuray on validation data')
        plt.xlabel('Values of lr')
        plt.title('mlp accuracy for different values of lr')
        plt.show()
        plt.close()
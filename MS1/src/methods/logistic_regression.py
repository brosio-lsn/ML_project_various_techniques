import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn, macrof1_fn

import matplotlib.pyplot as plt
from matplotlib import cm


class LogisticRegression(object):
    BEST_LR = 2e-3  # See report
    BEST_ITER = 180  # See report

    def visualize(self, training_data, training_labels, test_data, test_labels,
                  minLr, maxLr, minIter, maxIter, steps):
        """
            Prints the best accuracy found within the inferred set of parameters.
        Prints, along with the accuracy, the parameters used to find it (learning rate, max # of iter)
        Produces a plot of the accuracy of logistic regression in function of the learning rate
        and the maximum number of iterations.

        Arguments:
            training_data : the training data
            training_labels : the training labels
            test_data : the data to test the model (can be validation set)
            test_labels : the corresponding answer to the test data
            minLr : the minimum learning rate, default = 1e-5
            maxLr : the maximum learning rate, default = 5e-3
            minIter : the minimum maximum number of iterations, default = 50
            maxIter : the maximum maximum number of iterations, default = 500
            steps: the number of steps to go from minIter to maxIter, and minLr to maxLr
        Thus the method will fit steps² logistic regression models.
        Returns: void
        """
        ax = plt.axes(projection='3d')
        accuracies_train = np.zeros((steps, steps))
        accuracies_test = np.zeros((steps, steps))
        lr = np.linspace(minLr, maxLr, steps)
        max_iter = np.linspace(minIter, maxIter, steps).astype(int)
        X, Y = np.meshgrid(lr, max_iter)
        best_acc = 0.
        best_iter = 0.
        best_lr = 0.
        best_pred = 0
        print("===================================")
        print("VISUALIZING")
        print("===================================")
        for i in range(steps):
            for j in range(steps):
                print("In iteration", steps * i + j, "/", steps * steps - 1, " ...")
                obj = LogisticRegression(lr[i], max_iter[j])
                accuracies_train[i, j] = accuracy_fn(obj.fit(training_data, training_labels.reshape((-1, 1))),
                                                     training_labels)
                pred = obj.predict(test_data)
                accuracies_test[i, j] = accuracy_fn(obj.predict(test_data), test_labels)
                if accuracies_test[i, j] > best_acc:
                    best_iter = (maxIter - minIter) / steps * i
                    best_lr = (maxLr - minLr) / steps * i
                    best_acc = accuracies_test[i, j]
                    best_pred = pred
        print("Best accuracy :", best_acc)
        print("F1 Score : ", macrof1_fn(best_pred, test_labels))
        print("Learning rate used : ", best_lr)
        print("Max. number of iterations used : ", best_iter)
        ax.plot_surface(X, Y, accuracies_test, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Max. iterations')
        ax.set_zlabel('accuracy')
        plt.show()

    """
    Logistic regression classifier.
    """

    def f_softmax(self, data):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        expmatrix = np.exp(data @ self.weights)
        return expmatrix / np.sum(expmatrix, axis=1, keepdims=True)

    def gradient_logistic_multi(self, data, labels):
        """
    Compute the gradient of the entropy for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        W (array): Weights of shape (D, C)
    Returns:
        grad (np.array): Gradients of shape (D, C)
    """
        
        return data.T @ (self.f_softmax(data) - labels)

    def loss_logistic_multi(self, data, labels):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        y = self.f_softmax(data)

        return -np.sum(np.sum(labels * np.log(y)))

    def logistic_regression_predict_multi(self, data):
        """
        Prediction the label of data for multi-class logistic regression.

        Args:
            data (array): Dataset of shape (N, D).
        Returns:
            array of shape (N,): Label predictions of data.
        """
        y = self.f_softmax(data)
        label = onehot_to_label(y)
        return label

    def __init__(self, lr=BEST_LR, max_iters=BEST_ITER):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.weights = None
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        D = training_data.shape[1]  # number of features
        C = get_n_classes(training_labels)
        # number of classes

        self.weights = np.random.normal(0, 0.1, (D, C))
        # Random initialization of the weights
        labels_onehot = label_to_onehot(training_labels.squeeze(), C)

        for it in range(self.max_iters):
            gradient = self.gradient_logistic_multi(training_data, labels_onehot)
            self.weights = self.weights - self.lr * gradient
            predict = self.predict(training_data)
            if accuracy_fn(predict, training_labels) == 100:
                break
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        return self.logistic_regression_predict_multi(test_data)

"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC

# Optimal hyper-parameters found using cross-validation and finding extrema with matplotlib (code in svm_opt.py)
C_LIN = 0.006 
C_RBF, GAMMA_RBF = 3, 0.00075 
C_POLY, GAMMA_POLY, DEGREE_POLY, COEF0_POLY = 45, 1, 2, 750
OPT_KERNEL = 'rbf' 


class SVM(object):
    """
    SVM method.
    """

    def __init__(self, C=None, kernel=None, gamma=None, degree=None, coef0=None): 
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments. If parameters given in main.py (default are None) are default, set to optimal parameters.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """
        kernel = kernel if kernel is not None else OPT_KERNEL
        
        if kernel == 'linear':
            C = C if C is not None else C_LIN
            gamma = 'auto'
            degree = 0
            coef0 = 0
            
        elif kernel == 'poly':
            C = C if C is not None else C_POLY
            gamma = gamma if gamma is not None else GAMMA_POLY
            degree = degree if degree is not None else DEGREE_POLY
            coef0 = coef0 if coef0 is not None else COEF0_POLY
        elif kernel == 'rbf':
            C = C if C is not None else C_RBF
            gamma = gamma if gamma is not None else GAMMA_RBF
            degree = 0
            coef0 = 0

        self.classifier = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
            
        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.classifier.fit(training_data, training_labels)
        return self.predict(training_data)
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return self.classifier.predict(test_data)

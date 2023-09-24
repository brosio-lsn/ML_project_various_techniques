import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None
        self.eg = None
        self.Y = None


    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        self.mean = np.mean(training_data, axis=0)
        X_tilde = training_data - self.mean
        # Create the covariance matrix
        C = np.cov(X_tilde, rowvar=False)
        # Compute the eigenvectors and eigenvalues.
        eigvals, eigvecs = np.linalg.eigh(C)
        # Choose the top d eigenvalues and corresponding eigenvectors.
        indexes = np.argsort(eigvals)[::-1]
        self.W = eigvecs[:, indexes[:self.d]]
        self.eg = eigvals[indexes[:self.d]]

        return np.sum(self.eg)/np.sum(eigvals)*100

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        return (data-self.mean)@self.W

    def reconstruct_data(self, data):
        """
        Reconstruct the data after having reduced its dimensionality
        :param data: data to be reconstructed
        :return: reconstructed data
        """
        return self.mean + np.sum(data@self.W)
        


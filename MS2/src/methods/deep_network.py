import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.utils import accuracy_fn, label_to_onehot, macrof1_fn, get_n_classes
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm


## MS2

class MLP(nn.Module):
    """
    An MLP network which does classification.
    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, nbr_nodes=[512, 256, 128]):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            _init_(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.
        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        x = x.flatten(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class CNN(nn.Module):
    """
    A CNN which does classification.
    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, n1_filters=10, n2_filters=16, side=32):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.side = side
        self.n1_filters = n1_filters
        self.n2_filters = n2_filters
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.conv2d1 = nn.Conv2d(input_channels, n1_filters, kernel_size=3, padding=1)
        self.conv2d2 = nn.Conv2d(n1_filters, n2_filters, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(int(side/4) * int(side/4) * self.n2_filters, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.
        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###

        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        x = x.view(-1, int(self.side/4) * int(self.side/4) * self.n2_filters)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        ###
        ##
        return x


class Trainer(object):
    """
    Trainer class for the deep networks.
    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.
        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.n_classes = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.
        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        classes = self.n_classes
        if classes is None:
            classes=0
            for it, batch in enumerate(dataloader):
                x, y = batch
                classes = max(classes, get_n_classes(y.numpy()))
            self.n_classes=classes
        index = 0
        accuracies = []
        for ep in range(self.epochs):
            accuracies.append(self.train_one_epoch(dataloader, ep=index))
            index += 1
            if (ep % 10 == 0):
                print("Epoch: ", ep)



    def train_one_epoch(self, dataloader, ep=0):
        """
        Train the model for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!
        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        total_loss = 0.0
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch
            # print("x batch shape: ", x.shape, "y batch shape: ", y.shape)
            # turn y into one hot matrix
            y = y.numpy()
            y = label_to_onehot(y, self.n_classes)
            y = torch.from_numpy(y)
            # 5.2 Run forward pass.
            logits = self.model(x)

            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(logits, y)

            # 5.4 Run backward pass.
            loss.backward()
            total_loss += loss.item()

            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()

            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad()
        print("Average loss for this epoch : ", total_loss / (self.batch_size))
        return total_loss/self.batch_size
    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.
        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.
        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                x = batch[0]
                logits = self.model(x)
                pred = torch.argmax(logits, dim=1)
                pred_labels.append(pred)
        return torch.cat(pred_labels)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        This serves as an interface between numpy and pytorch.
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.n_classes = get_n_classes(training_labels)
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()


def visualize(model, training_data, training_labels, test_data, test_labels,
              lr, maxLr, minIter, maxIter, steps, batch_size=64):
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

    fig, ax1 = plt.subplots(1, 1)

    accuracies_train = np.zeros(steps)
    accuracies_test = np.zeros(steps)
    lr_tab = np.logspace(lr, maxLr, steps)

    best_acc_lr = 0.
    best_iter = 0.
    best_lr = 0.
    best_pred = 0
    print("===================================")
    print("VISUALIZING")
    print("===================================")
    trainers = []
    for i in range(steps):
        trainers.append(Trainer(model, lr_tab[i], minIter, batch_size))
    for i in range(steps):
        print("In iteration", i + 1, "/", steps, " with lr", lr_tab[i], "...")
        trainer = trainers[i]
        fit = trainer.fit(training_data, training_labels)
        accuracies_train[i] = accuracy_fn(fit,
                                          training_labels)

        pred = trainer.predict(test_data)
        accuracies_test[i] = accuracy_fn(pred, test_labels)
        print("Training accuracy : ", accuracies_train[i])
        print("Test accuracy : ", accuracies_test[i])
        if accuracies_test[i] > best_acc_lr:
            best_lr = lr_tab[i]
            best_acc_lr = accuracies_test[i]
            best_pred = pred
    print("Best accuracy for the learning rate:", best_acc_lr)
    print("F1 Score : ", macrof1_fn(best_pred, test_labels))
    print("Learning rate used : ", best_lr)
    print("Max. number of iterations used : ", best_iter)
    ax1.plot(lr_tab, accuracies_test, 'o-')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Learning rate')
    plt.show()
    max_iter = np.linspace(minIter, maxIter, steps).astype(int)

    plt.show()
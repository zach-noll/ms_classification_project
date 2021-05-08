import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################


#TODO: Hyperparameters that we need to test:
#      - minibatch size
#      - learning rate
#      - loss function
#      - activation function
#      - hidden layer size


########################################################################################################################


def load_dataset():
    # TODO: load the training and validation sets
    pass


def prepare_data():
    # TODO: prepare the data according to our network - divide the data into training set and validation set (20/80)??
    pass


def init_weights():
    # TODO: initialize the weights matrices, add biases (initialize these to 1??)
    pass


def forward_pass():
    # TODO: pass the minibatch through the NN and get the output. This will be called every iteration.
    pass


def calculate_loss():
    # TODO: calculate the loss according to desired loss function - can make this a switch case for different funcs.
    pass


def update_weights():
    # TODO: updates the weights matrices by backwards propogation and batch gradiant descent.
    pass

def run_epoch():
    # TODO: runs entire dataset through network in minibatches.
    pass

def display_results():
    # TODO: display the results from this run, will use this to tune hyperparameters
    pass


def main():
    return


if __name__ == "__main__":
    main()

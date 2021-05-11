import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math


########################################################################################################################


# TODO: Hyperparameters that we need to test:
#      - minibatch size
#      - learning rate
#      - loss function
#      - activation function
#      - hidden layer size
BATCH_SIZE= 16 #needs to be multiple of 2
LR = 0.01
LOSS_FUNC = 'BCE'
ACTIV_FUNC = 'sigmoid'
HIDDEN_LAYER = 100


########################################################################################################################
def normalize_image(image):
    """Min - max normalization of gray levels to 0-1 range."""
    min_value = image.min()
    max_value = image.max()
    normalized_im = (image - min_value) / (max_value - min_value)
    return normalized_im


def get_labels(image_name):
    """
    This function defines a numeric label to an image according to it's name, given as the input. A positive image
    gets numeric label of 1, and a negative image gets numeric label of 0
    """
    string_label = image_name.split('_')[0]
    if string_label == 'neg':
        label = 0
    elif string_label == 'pos':
        label = 1
    else:
        print("image name must contain 'pos' or 'neg' for defining image label")
    return label


def load_dataset(kind_of_set):
    """
    This function prepares the raw data set before entering to forward propagation stage.
    :param kind_of_set: indicates the name of the relevant data set, can be 'train' or 'validation'.
    :return: labels_list is a column vector which contains the label of each image in the raw data set.
             matrix_of_im_data is a matrix which it's columns contains the gray levels of each raw image in the input
             set after normalization to range 0-1, and flattening to a column vector.
    """
    relevant_folder_path = os.path.join('.', kind_of_set)
    list_of_files = os.listdir(relevant_folder_path)
    # Initialize outputs
    labels_list = np.zeros((len(list_of_files), 1))
    matrix_of_im_data = np.zeros((1024, len(list_of_files)))
    for idx, im_name in enumerate(list_of_files):
        im_path = os.path.join(relevant_folder_path, im_name)
        curr_im = cv2.imread(im_path, 0)
        curr_label = get_labels(im_name)
        if curr_label == 1:
            labels_list[idx, 0] = curr_label
        # Min-max normalization to 0-1 range
        normalized_im = normalize_image(curr_im)
        # Flattening image
        flattened_im = normalized_im.flatten(order='C')
        matrix_of_im_data[:, idx] = flattened_im.transpose()
    return labels_list.T, matrix_of_im_data.T


def prepare_data(data, labels):
    # TODO: prepare the data according to our network - concatenate the data and labels, then shuffle the samples
    """

    :param data: This is the data matrix from function load_dataset
    :param labels: This is the label vector that matches the data matrix
    :return: The matrix and label vector concatenated and shuffled
    """
    return


def init_weights(weight_dim, seed = 3):
    """
    Initializes the weight matrices
    :param weight_dim: dimensions are [number input features, number of hidden nodes, number of output nodes] in our case
    this is [512, H (hyperparameter), 1]
    :param seed: for random number generation, set default to 3
    :return: w1, w2, b1, b2 - weights and biases for layers 1 and 2
    """
    w1 = np.random.randn(weight_dim[1],weight_dim[0]) * np.sqrt(2/weight_dim[0])
    w2 = np.random.randn(weight_dim[2],weight_dim[1]) * np.sqrt(2/weight_dim[1])
    b1 = np.zeros((weight_dim[1],1))
    b2 = np.zeros((weight_dim[2],1))
    return w1, w2, b1, b2

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def feed_forward(X, w1, w2, b1, b2):
    z1 = w1 @ X.T + b1
    a1 = sigmoid(z1)

    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2


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

def train_NN(training_data,training_labels,w1,w2,b1,b2):
    epoch = 0
    num_of_batches = training_data.shape[0] // BATCH_SIZE
    while (1):
        epoch+=1

        for j in range(num_of_batches): #iterate over each mini batch

            for row in range(j*BATCH_SIZE,(j+1)*BATCH_SIZE): #iterate over each sample in mini batch
                X = training_data[row,:training_data.shape[1]] #this is the sample data
                #TODO: Y = trainind_data[row, -1] #this is the label
                #TODO: feed X, Y, w1, w2, b1, b2 into NN
                #TODO: initialize and sum the differentials

            # TODO: update weights and biases - W' = W - (1.0/N) * Del, N = batch size, del = diff vector from loss func.

        # TODO: print results for this epoch
        # TODO: check some stop condition (>90% accuracy)
    return w1, w2, b1, b2

def main():
    training_labels, training_data = load_dataset('training')
    val_labels, val_data = load_dataset('validation')
    num_of_features = training_data.shape[0]

    #initialize weights
    weight_dim = [num_of_features, HIDDEN_LAYER, 1] #[features, hidden, output]
    w1, w2, b1, b2 = init_weights(weight_dim)

    w1, w2, b1, b2 = train_NN(training_data,training_labels,w1,w2,b1,b2)


    #display results

    return


if __name__ == "__main__":
    main()

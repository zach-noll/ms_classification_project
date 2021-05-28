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
BATCH_SIZE = 4  # needs to be multiple of 2
LR = 0.01
LOSS_FUNC = 'BCE'
ACTIV_FUNC = 'sigmoid'
HIDDEN_LAYER = 15
EPOCHS = 300


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
    :return: pre_processed_data is a matrix which it's rows contains the gray levels of each raw image in the input
             set after normalization to range 0-1, and flattening to a column vector. The last column is a binary label
             of each image.
    """
    relevant_folder_path = os.path.join('.', kind_of_set)
    list_of_files = os.listdir(relevant_folder_path)
    # Initialize outputs
    labels_list = np.zeros((len(list_of_files), 1))
    matrix_of_im_data = np.zeros((len(list_of_files), 1024))
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
        matrix_of_im_data[idx, :] = flattened_im.transpose()
    pre_processed_data = np.concatenate((matrix_of_im_data, labels_list), axis=1)
    return pre_processed_data


def prepare_data(data_and_labels_matrix):
    """
    This function prepares the data according to our network by shuffling the samples
    :param data_and_labels_matrix: This is the output of function load_dataset
    :return: matrix of images values and labels after rows shuffling.
    """
    indices_to_take = np.random.rand(data_and_labels_matrix.shape[0]).argsort()
    shuffled_data_matrix = data_and_labels_matrix[indices_to_take, :]
    return shuffled_data_matrix


def init_weights(weight_dim, seed=3):
    """
    Initializes the weight matrices
    :param weight_dim: dimensions are [number input features, number of hidden nodes, number of output nodes] in our case
    this is [1024, H (hyperparameter), 1]
    :param seed: for random number generation, set default to 3
    :return: w1, w2, b1, b2 - weights and biases for layers 1 and 2
    """
    w1 = np.random.randn(weight_dim[1], weight_dim[0]) * np.sqrt(2 / weight_dim[0])
    w2 = np.random.randn(weight_dim[2], weight_dim[1]) * np.sqrt(2 / weight_dim[1])
    b1 = np.zeros((weight_dim[1], 1))
    b2 = np.zeros((weight_dim[2], 1))
    return w1, w2, b1, b2


def activation_func(activation_type, x, derivative=False):
    """
    :param activation_type: type of activation function: sigmoid, tanh of ReLU
    :param x: input array.
    :param derivative: a boolean variable -  whether to calculate the derivative of the desired activation function or not.
    :return: result of a desired activation function on the input x
    """
    if activation_type == 'sigmoid':
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))
    elif activation_type == 'tanh':
        if derivative:
            return 1 - (np.tanh(x) ** 2)
        return np.tanh(x)
    elif activation_type == 'ReLU':
        if derivative:
            return 1 * (x > 0)
        return x * (x > 0)  # Most efficient way
    else:
        print('Unrecognized activation function')


def feed_forward(X, w1, w2, b1, b2, activation_type):
    a1 = w1 @ X + b1
    z1 = activation_func(activation_type, a1, False)

    a2 = w2 @ z1 + b2
    z2 = activation_func(activation_type, a2, False)

    return np.atleast_2d(z1).T, np.atleast_2d(a1).T, np.atleast_2d(z2).T, np.atleast_2d(a2).T



# We will probably not need a function of the loss itself but only of the derivative of the loss
def calculate_loss(loss_func, label, pred):
    if loss_func == 'MSE':
        mse = (np.subtract(label, pred) ** 2)
        return mse

    elif loss_func == 'BCE':
        bce = -1 * label * np.log(pred) - (1 - label) * np.log(1 - pred)
        return bce



def calculate_loss_derivative(loss_func, label, pred):
    if loss_func == 'MSE':
        loss_derivative = label - pred

    elif loss_func == 'BCE':
        loss_derivative = -1 * label / pred + (1 - label) / (1 - pred)
    return loss_derivative


def update_weights():
    # TODO: updates the weights matrices by backwards propogation and batch gradiant descent.
    pass


def run_epoch():
    # TODO: runs entire dataset through network in minibatches.
    pass


def display_results():
    # TODO: display the results from this run, will use this to tune hyperparameters
    pass


def train_NN(training_data, w1, w2, b1, b2, activation_type, loss_type):

    num_of_batches = training_data.shape[0] // BATCH_SIZE

    for epoch in range (EPOCHS):


        for j in range(num_of_batches):  # iterate over each mini batch

            # Initilazing gradients
            delta_L = np.zeros((BATCH_SIZE,HIDDEN_LAYER))
            db2 = 0

            batch_accuracy = 0
            batch_loss = 0

            for row in range(j * BATCH_SIZE, (j + 1) * BATCH_SIZE):  # iterate over each sample in mini batch

                X = training_data[row, :-1]  # This is the sample data
                X = X.reshape((1024,1))

                Y = training_data[row, -1]  # This is the label


                # Feed forward
                a1 = w1 @ X + b1
                z1 = activation_func(ACTIV_FUNC, a1, False)

                a2 = w2 @ z1 + b2
                z2 = activation_func(ACTIV_FUNC, a2, False)

                # add loss for sample
                batch_loss += calculate_loss(LOSS_FUNC, Y, z2)

                output = np.round(a2)

                if Y == np.round(output):
                    batch_accuracy += 1

                # add gradients
                delta_L[row,:] += np.reshape(np.multiply(calculate_loss_derivative(LOSS_FUNC,Y,z2),activation_func(ACTIV_FUNC, a2, derivative=True)*z1),15)
                db2 += np.multiply(calculate_loss_derivative(LOSS_FUNC,Y,z2),activation_func(ACTIV_FUNC, a2, derivative=True))






            batch_loss = batch_loss / BATCH_SIZE
            batch_accuracy = batch_accuracy / BATCH_SIZE





            print(f"Average loss is: {batch_loss}")







        # TODO: print results for this epoch
        #print("")
       #print("[EPOCH #{}: Training accuracy: {}".format(epoch, loss))
        # TODO: check some stop condition (>90% accuracy)
    return w1, w2, b1, b2


def main():
    pre_processed_training = load_dataset('training')
    training_data = prepare_data(pre_processed_training)
    pre_processed_val = load_dataset('validation')
    val_data = prepare_data(pre_processed_training)
    pixels = training_data.shape[1]-1

    #initialize weights
    weight_dim = [pixels, HIDDEN_LAYER, 1] #[features, hidden, output]
    w1, w2, b1, b2 = init_weights(weight_dim)
    w1, w2, b1, b2 = train_NN(training_data, w1, w2, b1, b2,  'sigmoid', 'MSE')


    # display results

    return


if __name__ == "__main__":
    main()


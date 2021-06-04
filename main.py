import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
import json

########################################################################################################################


# Hyperparameters:
BATCH_SIZE = 2 # minibatch size, needs to be multiple of 2
LR = 0.01  # learning rate
LOSS_FUNC = 'BCE'  # loss function
ACTIV_FUNC = 'sigmoid'  # activation function
HIDDEN_LAYER = 50  # hidden layer size
EPOCHS = 60  # number of epoch

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
    This function prepares the data according to our network by shuffling the samples.
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
    b1 = np.ones((weight_dim[1], 1))
    b2 = np.ones((weight_dim[2], 1))
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
            return 1 - np.power(np.tanh(x), 2)
        return np.tanh(x)
    elif activation_type == 'relu':
        if derivative:
            return np.where(x > 0, 1.0, 0.0)
        return np.maximum(x, 0)

    else:
        print('Unrecognized activation function')


def feed_forward(X, w1, w2, b1, b2, activation_type):
    a1 = w1 @ X + b1
    z1 = activation_func(activation_type, a1, False)

    a2 = w2 @ z1 + b2
    z2 = activation_func(activation_type, a2, False)

    return a1, z1, a2, z2


def calculate_loss(loss_func, label, pred):
    if loss_func == 'MSE':
        mse = -0.5 * np.power(label - pred, 2)
        return mse

    elif loss_func == 'BCE':
        bce = -1 * label * np.log(pred) - (1 - label) * np.log(1 - pred)
        return bce


def calculate_loss_derivative(loss_func, label, pred):
    if loss_func == 'MSE':
        loss_derivative = -1 * (label - pred)

    elif loss_func == 'BCE':
        loss_derivative = -1 * label / pred + (1 - label) / (1 - pred)
    return loss_derivative


def calc_quality_indices(data, w1, w2, b1, b2, activation_type):
    """
    This function calculates vectors of loss and correct or incorrect predictions.
    """
    correct_vec = np.zeros((data.shape[0], 1))
    loss_vec = np.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):

        _, _, _, output_float = feed_forward(data[i, :-1].reshape(1024, 1), w1, w2, b1, b2, activation_type)

        loss_vec[i] = calculate_loss(LOSS_FUNC, data[i, -1], output_float)
        output = float(np.round(output_float))

        if (output == data[i, -1]):
            correct_vec[i] = 1

    return correct_vec, loss_vec


def plot_graphs(training_acc_arr,validation_acc_arr, training_loss_arr, validation_loss_arr):
    """plot graphs of accuraacy and loss for training and validation sets."""

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('# of epochs')
    ax1.set_ylabel('Accuracy [%]')
    ax1.plot(range(EPOCHS), training_acc_arr, label='Training')
    ax1.plot(range(EPOCHS), validation_acc_arr, label='Validation')
    ax1.set_ylim([40, 103])

    ax2.set_title('Loss')
    ax2.set_xlabel('# of epochs')
    ax2.set_ylabel('Loss')
    ax2.plot(range(EPOCHS), training_loss_arr, label='Training')
    ax2.plot(range(EPOCHS), validation_loss_arr, label='Validation')
    ax2.set_ylim([0, 0.8])

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()


def train_NN(training_data, validation_data, w1, w2, b1, b2, activation_type, loss_type):

    # Initializing per epoch accuracy and loss vectors
    training_acc_arr = np.zeros(EPOCHS)
    validation_acc_arr = np.zeros(EPOCHS)
    training_loss_arr = np.zeros(EPOCHS)
    validation_loss_arr = np.zeros(EPOCHS)

    num_of_batches = training_data.shape[0] // BATCH_SIZE

    for epoch in range(EPOCHS):

        for j in range(num_of_batches):  # iterate over each mini batch

            # Initializing gradients
            grad_E_L = np.zeros((HIDDEN_LAYER, 1))
            db2 = 0
            grad_E_H = np.zeros((HIDDEN_LAYER, 1024))
            db1 = np.zeros((HIDDEN_LAYER, 1))

            for row in range(j * BATCH_SIZE, (j + 1) * BATCH_SIZE):  # iterate over each sample in mini batch

                X = training_data[row, :-1]  # This is the sample data
                X = X.reshape((1024, 1))
                Y = training_data[row, -1]  # This is the label

                # Feed forward
                a1, z1, a2, z2 = feed_forward(X, w1, w2, b1, b2, activation_type)

                # add gradients
                # output layer
                del_L = calculate_loss_derivative(loss_type, Y, z2) * activation_func(activation_type, a2,
                                                                                      derivative=True)
                grad_E_L += del_L * z1
                db2 += del_L

                # hidden layer
                del_H = del_L * activation_func(activation_type, a1, derivative=True) * w2.T
                grad_E_H += del_H @ X.T
                db1 += del_H

            # update weights per minibatch using the average gradients over each minibatch

            w1 = w1 - LR * grad_E_H / BATCH_SIZE
            w2 = w2 - LR * grad_E_L.T / BATCH_SIZE
            b1 = b1 - LR * db1 / BATCH_SIZE
            b2 = b2 - LR * db2 / BATCH_SIZE

        training_output_vec, training_loss_vec = calc_quality_indices(training_data, w1, w2, b1, b2, activation_type)
        validation_output_vec, validation_loss_vec = calc_quality_indices(validation_data, w1, w2, b1, b2, activation_type)

        training_acc = np.around(np.average(training_output_vec) * 100, decimals=2)
        validation_acc = np.around(np.average(validation_output_vec) * 100, decimals=2)
        training_loss = np.around(np.average(training_loss_vec), decimals=2)
        validation_loss = np.around(np.average(validation_loss_vec), decimals=2)

        training_acc_arr[epoch] = training_acc
        validation_acc_arr[epoch] = validation_acc
        training_loss_arr[epoch] = training_loss
        validation_loss_arr[epoch] = validation_loss

        # Display results
        """
        print(f"[EPOCH] {epoch}: Training accuracy: {training_acc}%")
        print(f"           Validation accuracy: {validation_acc}%")
        print(f"           Training loss: {training_loss}")
        print(f"           Validation loss: {validation_loss}")
        """
        if epoch == (EPOCHS - 1):  # if last epoch
            print(f"Last epoch:\nTraining accuracy: {training_acc}%\nValidation accuracy: {validation_acc}%")
            print(f"Training loss: {training_loss}\nValidation loss: {validation_loss}")

    # Plot accuracy and loss graphs
    plot_graphs(training_acc_arr, validation_acc_arr, training_loss_arr, validation_loss_arr)

    return w1, w2, b1, b2


def make_json(W1, W2, b1, b2, id1, id2, activation1, activation2, nn_h_dim, path_to_save):
    """
    make json file with trained parameters.
    W1: numpy arrays of shape (1024, nn_h_dim)
    W2: numpy arrays of shape (nn_h_dim, 1)
    b1: numpy arrays of shape (1, nn_h_dim)
    b2: numpy arrays of shape (1, 1)
    nn_hdim - number of neirons in hidden layer: int
    id1: id1 - str '0123456789'
    id2: id2 - str '0123456789'
    activation1: one of only: 'sigmoid', 'tanh', 'ReLU'
    activation2: one of only: 'sigmoid', 'tanh', 'ReLU'
    """
    trained_dict = {'weights': (W1.tolist(), W2.tolist()),
                    'biases': (b1.tolist(), b2.tolist()),
                    'nn_hdim': nn_h_dim,
                    'activation_1': activation1,
                    'activation_2': activation2,
                    'IDs': (id1, id2)}
    file_path = os.path.join(path_to_save, 'trained_dict_{}_{}'.format(
        trained_dict.get('IDs')[0], trained_dict.get('IDs')[1])
                             )
    with open(file_path, 'w') as f:
        json.dump(trained_dict, f, indent=4)


def get_ids_from_file(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def main():
    pre_processed_training = load_dataset('training')
    training_data = prepare_data(pre_processed_training)
    pre_processed_val = load_dataset('validation')
    val_data = prepare_data(pre_processed_val)
    pixels = training_data.shape[1] - 1

    # initialize weights
    weight_dim = [pixels, HIDDEN_LAYER, 1]  # [features, hidden, output]
    w1, w2, b1, b2 = init_weights(weight_dim)

    # Train the NN, predict on validation and display results
    w1, w2, b1, b2 = train_NN(training_data, val_data, w1, w2, b1, b2, ACTIV_FUNC, LOSS_FUNC)

    id0, id1 = get_ids_from_file("ids.txt")

    # Save to json
    make_json(w1.T, w2.T, b1.T, b2, id0, id1, ACTIV_FUNC, ACTIV_FUNC, HIDDEN_LAYER, os.getcwd())


if __name__ == "__main__":
    main()

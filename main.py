import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

########################################################################################################################


#TODO: Hyperparameters that we need to test:
#      - minibatch size
#      - learning rate
#      - loss function
#      - activation function
#      - hidden layer size


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
    return labels_list, matrix_of_im_data


def prepare_data():
    # TODO: prepare the data according to our network - divide the data into training set and validation set (20/80)??
    # we don't need that function
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
    training_labels, training_data = load_dataset('training')
    val_labels, val_data = load_dataset('validation')

    #initialize weights

    #for i in number of epochs

        #for j in number of batches

            #for k in number of samples in batch j

                #feed forward sample k through network
                #sum the differential

            #update weights and biases - W' = W - (1.0/N) * Del, N = batch size, del = diff vector from loss func.

        #print results for this epoch
        #check some stop condition (>90% accuracy)

    #display results

    return


if __name__ == "__main__":
    main()

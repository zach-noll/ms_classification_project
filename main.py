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
    # Min - max normalization to 0-1 range.
    min_value = image.min()
    max_value = image.max()
    normalized_im = (image - min_value) / (max_value - min_value)
    return normalized_im


def get_label(image_name):

    return label


def faltten_image():

    return flattened_im


def load_dataset(kind_of_set):
    # TODO: load the training and validation sets

    # Get the path of current working directory
    curr_path = os.getcwd()
    relevant_folder_path = os.path.join('.', kind_of_set)
    for img_name in os.listdir(relevant_folder_path):
        img_path = os.path.join(relevant_folder_path, img_name)
        curr_img = cv2.imread(img_path, 0)
        # Min-max normalization to 0-1 range
        normalized_im = normalize_image(image)
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
    load_dataset('training')
    return


if __name__ == "__main__":
    main()

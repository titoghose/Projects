import os
import sys

import cv2

from data_preprocessing import change_shape, normalized, prep_data
from segnet_model import create_model
import numpy as np
import matplotlib.pyplot as plt

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road_marking = [255, 69, 0]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

segnet_model = create_model()


def visualize(img):

    r = img.copy()
    g = img.copy()
    b = img.copy()
    for l in range(0, 11):
        r[img == l] = label_colours[l, 0]
        g[img == l] = label_colours[l, 1]
        b[img == l] = label_colours[l, 2]

    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)  # [:,:,0]
    rgb[:, :, 1] = (g / 255.0)  # [:,:,1]
    rgb[:, :, 2] = (b / 255.0)  # [:,:,2]

    return rgb


def test(img):
    img = change_shape(img)
    img = normalized(img)

    #output = segnet_model.predict(img)
    #pred = visualize(np.argmax(output[0], axis=1).reshape(360, 480))
    return img


#img_path = sys.argv[1]
#test(cv2.imread(img_path))
'''
test_data,test_labels = prep_data('test')
test_labels = np.reshape(test_labels, (-1, 360 * 480, 12))
for i in range(len(test_data)):
    test(test_data[i])
'''
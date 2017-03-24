# Pre-process the data before training or testing

import os
import numpy as np
import cv2


def normalized(rgb):
    # return rgb/255.0
    norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

    b = rgb[:, :, 0]
    g = rgb[:, :, 1]
    r = rgb[:, :, 2]

    norm[:, :, 0] = cv2.equalizeHist(b)
    norm[:, :, 1] = cv2.equalizeHist(g)
    norm[:, :, 2] = cv2.equalizeHist(r)

    return norm


def one_hot_encode(labels):
    x = np.zeros([360, 480, 12])
    for i in range(360):
        for j in range(480):
            x[i, j, labels[i][j]] = 1
    return x


def prep_data(name):
    with open('CamVid/' + name + '.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]

    data = []
    labels = []
    for i in range(len(txt)):
        data.append(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
        labels.append(one_hot_encode(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:, :, 0]))

    return np.array(data), np.array(labels)


def change_shape(img):
    res = cv2.resize(img, None, fx=(480. / img.shape[1]), fy=(360. / img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return res

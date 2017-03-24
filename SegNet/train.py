import os
import numpy as np
from data_preprocessing import prep_data
from segnet_model import create_model
from multi_gpu import make_parallel
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

segnet_model = create_model()
segnet_model.summary()
print "Model Created"

train_data, train_labels = prep_data('train')
train_labels = np.reshape(train_labels, (-1, 360 * 480, 12))
# train_data = train_data[79:80]
# train_labels = train_labels[79:80]
print "Data pre processed"
print "Training data: ", train_data.shape
print "Training Labels: ", train_labels.shape


sgd = SGD(lr=0.1, momentum=0.9)
# segnet_model = make_parallel(segnet_model, 2)
segnet_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print "Model compiled"


class_weights = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

path = "weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

val_data, val_labels = prep_data('val')
val_labels = np.reshape(val_labels, (-1, 360 * 480, 12))

#segnet_model.load_weights("/home/ws1/Desktop/SegNet/weights/weights-improvement-375-0.1603.hdf5")

segnet_model.fit(train_data, train_labels, nb_epoch=2000, batch_size=1, shuffle=True,
                 class_weight=class_weights, verbose=1, callbacks=callbacks_list)

# Basic SegNet Model

from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


def create_model():
    input_layer = Input(shape=(360, 480, 3))

    # Encoder network
    encoder1 = ZeroPadding2D(padding=(1, 1))(input_layer)
    encoder1 = Conv2D(64, 3, 3)(encoder1)
    encoder1 = BatchNormalization()(encoder1)
    encoder1 = Activation('relu')(encoder1)
    encoder1 = ZeroPadding2D(padding=(1, 1))(encoder1)
    encoder1 = Conv2D(64, 3, 3)(encoder1)
    encoder1 = BatchNormalization()(encoder1)
    encoder1 = Activation('relu')(encoder1)
    encoder1 = merge([encoder1, input_layer], mode='sum')
    encoder1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder1)

    encoder2 = ZeroPadding2D(padding=(1, 1))(encoder1)
    encoder2 = Conv2D(64, 3, 3)(encoder2)
    encoder2 = BatchNormalization()(encoder2)
    encoder2 = Activation('relu')(encoder2)
    encoder2 = ZeroPadding2D(padding=(1, 1))(encoder2)
    encoder2 = Conv2D(64, 3, 3)(encoder2)
    encoder2 = BatchNormalization()(encoder2)
    encoder2 = Activation('relu')(encoder2)
    encoder2 = merge([encoder2, encoder1], mode='sum')
    encoder2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder2)

    encoder3 = ZeroPadding2D(padding=(1, 1))(encoder2)
    encoder3 = Conv2D(64, 3, 3)(encoder3)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = Activation('relu')(encoder3)
    encoder3 = ZeroPadding2D(padding=(1, 1))(encoder3)
    encoder3 = Conv2D(64, 3, 3)(encoder3)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = Activation('relu')(encoder3)
    encoder3 = ZeroPadding2D(padding=(1, 1))(encoder3)
    encoder3 = Conv2D(64, 3, 3)(encoder3)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = Activation('relu')(encoder3)
    encoder3 = merge([encoder3, encoder2], mode='sum')
    encoder3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder3)

    encoder4 = ZeroPadding2D(padding=(1, 1))(encoder3)
    encoder4 = Conv2D(64, 3, 3)(encoder4)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = Activation('relu')(encoder4)
    encoder4 = ZeroPadding2D(padding=(1, 1))(encoder4)
    encoder4 = Conv2D(64, 3, 3)(encoder4)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = Activation('relu')(encoder4)
    encoder4 = ZeroPadding2D(padding=(1, 1))(encoder4)
    encoder4 = Conv2D(64, 3, 3)(encoder4)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = Activation('relu')(encoder4)
    encoder4 = merge([encoder4, encoder3], mode='sum')
    encoder4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder4)

    # Decoder network
    decoder1 = UpSampling2D(size=(2, 2))(encoder4)
    decoder1 = ZeroPadding2D(padding=(1, 1))(decoder1)
    decoder1 = Conv2D(64, 3, 3)(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = ZeroPadding2D(padding=(1, 1))(decoder1)
    decoder1 = Conv2D(64, 3, 3)(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = ZeroPadding2D(padding=(1, 1))(decoder1)
    decoder1 = Conv2D(64, 3, 3)(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = ZeroPadding2D(padding=(1, 0, 0, 0))(decoder1)

    decoder2 = UpSampling2D(size=(2, 2))(decoder1)
    decoder2 = ZeroPadding2D(padding=(1, 1))(decoder2)
    decoder2 = Conv2D(64, 3, 3)(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = ZeroPadding2D(padding=(1, 1))(decoder2)
    decoder2 = Conv2D(64, 3, 3)(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = ZeroPadding2D(padding=(1, 1))(decoder2)
    decoder2 = Conv2D(64, 3, 3)(decoder2)
    decoder2 = BatchNormalization()(decoder2)

    decoder3 = UpSampling2D(size=(2, 2))(decoder2)
    decoder3 = ZeroPadding2D(padding=(1, 1))(decoder3)
    decoder3 = Conv2D(64, 3, 3)(decoder3)
    decoder3 = BatchNormalization()(decoder3)
    decoder3 = ZeroPadding2D(padding=(1, 1))(decoder3)
    decoder3 = Conv2D(64, 3, 3)(decoder3)
    decoder3 = BatchNormalization()(decoder3)

    decoder4 = UpSampling2D(size=(2, 2))(decoder3)
    decoder4 = ZeroPadding2D(padding=(1, 1))(decoder4)
    decoder4 = Conv2D(64, 3, 3)(decoder4)
    decoder4 = BatchNormalization()(decoder4)
    decoder4 = ZeroPadding2D(padding=(1, 1))(decoder4)
    decoder4 = Conv2D(64, 3, 3)(decoder4)
    decoder4 = BatchNormalization()(decoder4)

    classification = Conv2D(12, 1, 1)(decoder4)
    classification = Reshape((360 * 480, 12), input_shape=(360, 480, 12))(classification)
    classification = Activation('softmax')(classification)

    # creating autoencoder
    autoencoder = Model(input=input_layer, output=classification)

    return autoencoder

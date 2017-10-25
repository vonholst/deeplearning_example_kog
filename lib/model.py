# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.constraints import maxnorm


def keras_model(input_shape):
    model = Sequential()
    model.add(Conv2D(8, (5, 5), padding='valid', input_shape=input_shape))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(240))

    model.add(Activation('relu'))
    model.add(Dense(120))

    #model.add(Activation('relu'))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    return model


def old_model(input_shape):
    # Define model architecture
    # 1x100 -> (3) 32x100 ,(3) 32x100, [4] 25, (3) 25, (3) 25, [2] 12, (3) 12, (3) 12, [2] 6, FC...
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

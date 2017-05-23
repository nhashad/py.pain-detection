import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.advanced_activations import PReLU

NUM_CLASSES = 7
PICTURE_DIM = 48

def print_metadata(x_train, x_test):
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


def build_model(x_train):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    #model.add(PReLU(alpha_initializer='zeros'))
    
    
    model.add(Flatten())
    
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    return model
    
"""model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(Activation('relu'))"""

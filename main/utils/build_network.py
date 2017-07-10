import keras
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import LSTM,TimeDistributed
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, ZeroPadding2D
from keras.layers.merge import concatenate

NUM_CLASSES = 8
NUM_CLASSES_PAIN=13
PICTURE_DIM = 48


import sys
sys.setrecursionlimit(10000)


def BNConv(nb_filter, nb_row, nb_col, w_decay, padding="same"):
    def f(input):
        conv = Conv2D(nb_filter, (nb_row, nb_col), padding=padding, activation="relu",
                     kernel_initializer="he_normal", kernel_regularizer=None)(input)
        return BatchNormalization(axis=1)(conv)
    return f


def inception_v3(w_decay=None):
    input = Input(shape=(48, 48, 1))

    conv_1 = BNConv(32, 3, 3, w_decay, padding="valid")(input)
    conv_2 = BNConv(32, 3, 3, w_decay, padding="valid")(conv_1)
    pool_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv_2)

    conv_4 = BNConv(80, 1, 1, w_decay)(pool_3)
    conv_5 = BNConv(92, 3, 3, w_decay, padding="valid")(conv_4)
    pool_6 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv_5)

    inception_7 = InceptionFig5(w_decay)(pool_6)

    inception_8 = DimReductionA(w_decay)(inception_7)

    inception_9 = InceptionFig6(w_decay)(inception_8)

    inception_10 = DimReductionB(w_decay)(inception_9)

    inception_11 = InceptionFig7(w_decay)(inception_10)    

    pool_13 = Lambda(lambda x: K.mean(x, axis=(1, 2)))(inception_12)
    
    pool_13 = Dense(NUM_CLASSES, activation= 'softmax')(pool_13)

    model = Model(input, pool_13)

    return model


def InceptionFig5(w_decay):
    def f(input):

        # Tower A
        conv_a1 = BNConv(64, 1, 1, w_decay)(input)
        conv_a2 = BNConv(96, 3, 3, w_decay)(conv_a1)
        conv_a3 = BNConv(96, 3, 3, w_decay)(conv_a2)

        # Tower B
        conv_b1 = BNConv(48, 1, 1, w_decay)(input)
        conv_b2 = BNConv(64, 3, 3, w_decay)(conv_b1)

        # Tower C
        pool_c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
        conv_c2 = BNConv(64, 1, 1, w_decay)(pool_c1)

        # Tower D
        conv_d1 = BNConv(64, 1, 1, w_decay)(input)

        return concatenate([conv_a3, conv_b2, conv_c2, conv_d1], axis=3)

    return f


def InceptionFig6(w_decay):
    def f(input):
        # Tower A
        conv_a1 = BNConv(90, 1, 1, w_decay)(input)
        conv_a2 = BNConv(90, 1, 7, w_decay)(conv_a1)
        conv_a3 = BNConv(90, 7, 1, w_decay)(conv_a2)

        # Tower B
        conv_b1 = BNConv(64, 1, 1, w_decay)(input)
        conv_b2 = BNConv(64, 1, 7, w_decay)(conv_b1)
        conv_b3 = BNConv(92, 7, 1, w_decay)(conv_b2)

        # Tower C
        pool_c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
        conv_c2 = BNConv(92, 1, 1, w_decay)(pool_c1)

        # Tower D
        conv_d = BNConv(92, 1, 1, w_decay)(input)

        return concatenate([conv_a3, conv_b3, conv_c2, conv_d], axis=3)

    return f


def InceptionFig7(w_decay):
    def f(input):
        # Tower A
        conv_a1 = BNConv(192, 1, 1, w_decay)(input)
        conv_a2 = BNConv(184, 3, 3, w_decay)(conv_a1)

        # Tower B
        conv_b1 = BNConv(184, 1, 1, w_decay)(input)
        conv_b2 = BNConv(184, 1, 3, w_decay)(conv_b1)
        conv_b3 = BNConv(184, 3, 1, w_decay)(conv_b2)

        # Tower C
        pool_c1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input)
        conv_c2 = BNConv(92, 1, 1, w_decay)(pool_c1)

        # Tower D
        conv_d = BNConv(120, 1, 1, w_decay)(input)

        return concatenate([conv_a2, conv_b3, conv_c2, conv_d], axis=3)

    return f


def DimReductionA(w_decay):
    def f(input):
        # Tower A
        conv_a1 = BNConv(64, 1, 1, w_decay)(input)
        conv_a2 = BNConv(96, 3, 3, w_decay)(conv_a1)
        conv_a3 = BNConv(96, 3, 3, w_decay, padding="valid")(conv_a2)
        
        # Tower B
        conv_b = BNConv(184, 3, 3, w_decay, padding="valid")(input)

        # Tower C
        pool_c = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="valid")(input)

        return concatenate([conv_a3, conv_b, pool_c], axis=3)
    return f


def DimReductionB(w_decay):
    def f(input):
        # Tower A
        conv_a1 = BNConv(92, 1, 1, w_decay)(input)
        conv_a2 = BNConv(120, 3, 3, w_decay, padding="valid")(conv_a1)

        # Tower B
        conv_b1 = BNConv(92, 1, 1, w_decay)(input)
        conv_b2 = BNConv(92, 3, 3, w_decay, padding="valid")(conv_b1)

        # Tower C
        pool_c = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="valid")(input)

        return concatenate([conv_a2, conv_b2, pool_c], axis=3)
    return f



def print_metadata(x_train, x_test):
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


def build_model(x_train):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
        
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu')) 
    
    model.add(Flatten())
    
    if (x_train.shape[1] == 160):
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(600))
        model.add(Activation('relu'))
        
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    return model


def build_pain_model(x_train):
    
    model = Sequential()
    
    model.add(Conv2D(32, (10, 10, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (10, 10, 3), padding='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(4, 4)))
    
    model.add(Conv2D(32, (10, 10, 3), padding='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(4, 4)))
    
    model.add(Flatten())
              
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(600))
    model.add(Activation('relu'))
        
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(NUM_CLASSES_PAIN))
    model.add(Activation('softmax'))

    
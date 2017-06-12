import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from datetime import datetime
import time
import pickle
import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# reset everything to rerun in jupyter
tf.reset_default_graph()

batch_size = 128
num__of_classes = 8
epochs = 20

layer1_size = 32

def train_model(train_file='fer2013.pickle', job_dir='./tmp/main-1', **args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_file))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')
    file_stream = file_io.FileIO(train_file, mode='r')
    x_train, y_train, x_test, y_test, x_val, y_val = pickle.load(file_stream)
    
    print(x_train.shape, y_train.shape, 'train samples,', type(x_train[0][0]), ' ', type(y_train[0][0]))
    print(x_val.shape, y_val.shape, 'validation samples,', type(x_val[0][0]), ' ', type(y_val[0][0]))
    print(x_test.shape,  y_test.shape,  'test samples,',  type(x_test[0][0]),  ' ', type(y_test[0][0]))

    # convert class vectors to binary class matrices. Our input already made this. No need to do it again
    y_train, y_test, y_val = keras.utils.to_categorical(y_train, num_of_classes), keras.utils.to_categorical(y_test, num_of_classes),keras.utils.to_categorical(y_val, num_of_classes)

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
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    
    opt = keras.optimizers.Adagrad(lr= 0.01, epsilon= 1e-08,  decay= 0.0)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
             validation_data= (x_val, y_val), shuffle=True, verbose=1)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('model.h5')
    
    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    
    train_model(**arguments)
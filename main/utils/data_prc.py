import os
import pickle
import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils

TRAIN_SIZE = 28709
DATASET_SIZE = 35887
NUM_CLASSES = 6
PICTURE_DIM = 48

DATASET_PATH = "../datasets/emotions/fer2013/"

dataset = np.zeros((DATASET_SIZE,3))

def get_dataset():
    return dataset

def remove_disgust(dataset):
    emotion = dataset.pop('emotion')
    print ("Changing Disgust to Anger")
    
    for i in range(emotion.size):
        if(emotion[i] == 0 or emotion[i] == 1):
            emotion[i] = 0
        else:
            emotion[i] -= 1
    
    dataset['emotion'] = emotion
    return dataset
    

def dataset_pickle(filename, force):

    filename = DATASET_PATH + filename

    pickle_file  = os.path.splitext(filename)[0] + '.pickle'

    global dataset
    if(os.path.exists(pickle_file) and not force):
        print ('%s already exists. Skipping pickling.' % pickle_file)
        dataset = pd.read_csv(filename)
        dataset = remove_disgust(dataset)
    else:
        with open(filename, 'rb') :
            dataset = pd.read_csv(filename)
            dataset = remove_disgust(dataset)
            X_train = dataset.pixels[0:TRAIN_SIZE]
            y_train = dataset.emotion[0:TRAIN_SIZE]

            X_test = dataset.pixels[TRAIN_SIZE:DATASET_SIZE]
            y_test = dataset.emotion[TRAIN_SIZE:DATASET_SIZE]

        
        X_train = np.array(list(map(lambda arr: np.fromiter(list(map(lambda str: int(str),
                     arr)), dtype= np.int), list(map(lambda str: str.split(),
                      X_train)))))

        y_train = np.fromiter(list(map(int, y_train)), dtype=np.int)

        X_test = np.array(list(map(lambda arr: np.fromiter(list(map(lambda str: int(str),
                     arr)), dtype= np.int), list(map(lambda str: str.split(),
                      X_test)))))
        y_test = np.fromiter(list(map(int, y_test)), dtype=np.int)
        
        
        print ('Pickling', pickle_file, '...')

        with open(pickle_file, 'wb') as picklefile:
            save = {
                'dataset_Xtrain': X_train,
                'dataset_ytrain': y_train,
                'dataset_Xtest': X_test,
                'dataset_ytest': y_test
            }
            pickle.dump(save, picklefile, pickle.HIGHEST_PROTOCOL)
            print (pickle_file, 'pickled successfully!')
          
        

def dataset_loading(filename):

    filename = DATASET_PATH + filename + '.pickle'

    with open(filename, 'rb') as picklefile:
        save = pickle.load(picklefile)

        X_train = save['dataset_Xtrain']

        y_train = save['dataset_ytrain']

        X_test = save['dataset_Xtest']

        y_test = save['dataset_ytest']

    return X_train, y_train, X_test, y_test

def prepare_examples(x_train, x_test):
    
    return np.reshape(x_train,(x_train.shape[0], PICTURE_DIM, PICTURE_DIM,1)),np.reshape(x_test,(x_test.shape[0], PICTURE_DIM, PICTURE_DIM,1))

def y_to_categorical(y_train, y_test):
    
    return keras.utils.to_categorical(y_train, NUM_CLASSES), keras.utils.to_categorical(y_test, NUM_CLASSES)

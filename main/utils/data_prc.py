import os, os.path
import pickle
import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as image_utils
from PIL import Image
from scipy.misc import imread, imsave



#emotion
TRAIN_SIZE = 28709
DATASET_SIZE = 35887
NUM_CLASSES = 7
PICTURE_DIM = 48
DATASET_PATH = "../datasets/emotions/fer2013/"

#pain
TRAIN_SIZE_PAIN = 26859
TRAIN_SIZE_PAIN_GEATER_2 = 1873
VALIDATION_SIZE_PAIN = 9541
VALIDATION_SIZE_PAIN_GEATER_2 = 586
TEST_SIZE_PAIN = 9753
TEST_SIZE_PAIN_GEATER_2 = 652
NUM_CLASSES_PAIN = 16
PICTURE_DIM_PAIN_H = 240
PICTURE_DIM_PAIN_W = 320
DATASET_PATH_PAIN = "../datasets/pain/pain_organized_ds/"

dataset = np.zeros((DATASET_SIZE,3))

def dataset_pickle_pain(filename):
    
    pickle_file  = DATASET_PATH_PAIN + filename+'.pickle'
    if(os.path.exists(pickle_file)):
        print ('%s already exists. Skipping pickling.' % pickle_file)
    
    else:
        X_train = np.empty([TRAIN_SIZE_PAIN_GEATER_2, PICTURE_DIM_PAIN_H,PICTURE_DIM_PAIN_W,3])
        y_train = np.empty([TRAIN_SIZE_PAIN_GEATER_2,])
        X_val = np.empty([VALIDATION_SIZE_PAIN_GEATER_2, PICTURE_DIM_PAIN_H,PICTURE_DIM_PAIN_W,3])
        y_val = np.empty([VALIDATION_SIZE_PAIN_GEATER_2,])
        X_test = np.empty([TEST_SIZE_PAIN_GEATER_2, PICTURE_DIM_PAIN_H,PICTURE_DIM_PAIN_W,3])
        y_test = np.empty([TEST_SIZE_PAIN_GEATER_2,])

        train_indx = 0
        val_indx = 0 
        test_indx = 0
        for label,folder_name in enumerate(os.listdir(DATASET_PATH_PAIN)):
            folder = os.path.join(DATASET_PATH_PAIN,folder_name)
            print(label,folder_name)
            for lbl, subfolder_name in enumerate(os.listdir(folder)):
                subfolder = os.path.join(folder, subfolder_name)
                if (int(subfolder_name) > 2):
                    for f in os.listdir(subfolder):
                        fileName = os.path.join(subfolder, f)
                        image = image_utils.load_img(fileName, target_size=(PICTURE_DIM_PAIN_H, PICTURE_DIM_PAIN_W))
                        image = image_utils.img_to_array(image).astype(np.float32)
                        image = image/ 255.

                        #print ('Size: ', image.size)
                        #print ('Shape: ', image.shape)
                        #plt.imshow(image)
                        #plt.show()
                        #print(image)

                        if (folder_name == 'train'):
                            X_train[train_indx] = image
                            y_train[train_indx] = int(subfolder_name)
                            train_indx +=1
                        elif(folder_name == 'test'):
                            X_test[test_indx] = image
                            y_test[test_indx] = int(subfolder_name)
                            test_indx +=1
                        else:
                            X_val[val_indx] = image
                            y_val[val_indx] = int(subfolder_name)
                            val_indx +=1


        rand_train =  np.arange(TRAIN_SIZE_PAIN_GEATER_2)
        np.random.shuffle(rand_train)
        y_train = y_train[rand_train]
        X_train = X_train[rand_train]

        rand_test =  np.arange(TEST_SIZE_PAIN_GEATER_2)
        np.random.shuffle(rand_test)
        y_test = y_test[rand_test]
        X_test = X_test[rand_test]

        rand_val =  np.arange(VALIDATION_SIZE_PAIN_GEATER_2)
        np.random.shuffle(rand_val)
        y_val = y_val[rand_val]
        X_val = X_val[rand_val]
        
        print ('Pickling', pickle_file, '...')

        with open(pickle_file, 'wb') as picklefile:
            save = {
                'dataset_Xtrain': X_train,
                'dataset_ytrain': y_train,
                'dataset_Xtest': X_test,
                'dataset_ytest': y_test,
                'dataset_Xval': X_val,
                'dataset_yval': y_val
            }
            pickle.dump(save, picklefile, pickle.HIGHEST_PROTOCOL)
            print (pickle_file, 'pickled successfully!')

def dataset_loading(filename):

    filename = DATASET_PATH_PAIN + filename + '.pickle'

    with open(filename, 'rb') as picklefile:
        save = pickle.load(picklefile)

        X_train = save['dataset_Xtrain']

        y_train = save['dataset_ytrain']

        X_test = save['dataset_Xtest']

        y_test = save['dataset_ytest']
        
        X_val = save['dataset_Xval']

        y_val = save['dataset_yval']

    return X_train, y_train, X_test, y_test,  X_val, y_val
            
                

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
        #dataset = remove_disgust(dataset)
    else:
        with open(filename, 'rb') :
            dataset = pd.read_csv(filename)
            #dataset = remove_disgust(dataset)
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
    
    x_train, x_test =  np.reshape(x_train,(x_train.shape[0], PICTURE_DIM, PICTURE_DIM,1)),np.reshape(x_test,(x_test.shape[0], PICTURE_DIM, PICTURE_DIM,1))
    
    x_train = x_train.astype('float32')
    x_train/=255
     
    x_test = x_test.astype('float32')
    x_test/=255
    
    return x_train, x_test

def y_to_categorical(y_train, y_test):
    
    return keras.utils.to_categorical(y_train, NUM_CLASSES), keras.utils.to_categorical(y_test, NUM_CLASSES)

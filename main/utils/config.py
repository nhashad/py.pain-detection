#Imports
import os, os.path
import pickle
import keras
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.preprocessing import image as image_utils
from PIL import Image
from scipy.misc import imread, imsave
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#emotion metadata
TRAIN_SIZE_EMOTION = 20097
VALIDATION_SIZE_EMOTION = 8612 #0.3*28709
DATASET_SIZE_EMOTION = 35887
NUM_CLASSES_EMOTION = 7
PICTURE_DIM_EMOTION = 48
DATASET_PATH_EMOTION = "../datasets/emotions/fer2013/"
DATASET_NAME_EMOTION = "fer2013"

FIN_TRAIN_SIZE_EMOTION = 21970
FIN_VALIDATION_SIZE_EMOTION = 9198
FIN_TEST_SIZE_EMOTION = 7830


#pain metadata
TRAIN_SIZE_PAIN = 26859
TRAIN_SIZE_PAIN_GEATER_2 = 1873
VALIDATION_SIZE_PAIN = 9541
VALIDATION_SIZE_PAIN_GEATER_2 = 586
TEST_SIZE_PAIN = 9753
TEST_SIZE_PAIN_GEATER_2 = 652
NUM_CLASSES_PAIN = 16
PICTURE_DIM_PAIN_H = 160
PICTURE_DIM_PAIN_W = 160
DATASET_PATH_PAIN = "../datasets/pain/pain_organized_ds/"
DATASET_NAME_PAIN = "pain_ds"


#sr metadata
DATASET_PATH_PAIN_GSR = "../datasets/pain/"
DATASET_SIZE_GSR = 8500
TRAIN_SIZE_GSR = 5950
VALIDATION_SIZE_GSR = 1690
TEST_SIZE_GSR = 843
ALL_FEATURES_NUM = 156
SR_CARDIOLOGY_FEATURE_NUM = 51
SR_FEATURE_NUM = 3
DATASET_NAME_SR = "GSR_ds"


NUM_CLASSES_EMOTION = 8
NUM_CLASSES_PAIN=13
NUM_CLASSES_SR = 5
PICTURE_DIM = 48
PIC_DIM_PAIN = 160

emotion = { 0:'Angry', 1:'Disgust', 2: 'Fear', 3: 'Happy',
           4: 'Sad', 5: 'Surprise', 6: 'Neutral', 7: 'Pain'}
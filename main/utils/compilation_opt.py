import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score
from sklearn import svm



def sgdOpt(learningrate= 0.01, momentum= 0.0,  decay= 0.0, nestrov=False):
    
    opt = SGD(lr= learningrate, momentum=momentum, decay=decay, nesterov=nestrov)
    
    return opt
    

def rmsPropOpt(learningrate=0.00025, rho=0.9, epsilon=1e-08, decay=0.0):
    
    opt = keras.optimizers.rmsprop(lr = learningrate, rho=rho, epsilon= epsilon, decay = decay)
        
    return opt


def adagradOpt(learningrate= 0.01, epsilon= 1e-08,  decay= 0.0):
    
    opt = keras.optimizers.Adagrad(lr= learningrate, epsilon= epsilon, decay= decay)

    return opt

def adamOpt(learningrate= 0.01, epsilon= 1e-08,  decay= 0.0):

    opt = keras.optimizers.Adam(lr= learningrate, beta_1=0.9, beta_2=0.999, epsilon= epsilon, decay= decay)
    
    return opt

def compiling(model, opt):
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model
 
def training_cross_valid(model, batch_size, epochs, x_train, y_train):
    
    # define 10-fold cross validation test harness
    kf = KFold(n_splits=10, shuffle=True)
    cvscores = []
    for train, test in kf.split(x_train):
        # Fit the model
        model.fit(x_train[train], y_train[train], epochs=epochs, batch_size=batch_size, verbose=0)
        # evaluate the model
        scores = model.evaluate(x_train[test], y_train[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %f" % (model.metrics_names[0], scores[0]))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return model

    
    
def training(model, batch_size, epochs, x_train, y_train, x_val, y_val):
    
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
        validation_data= (x_val, y_val), shuffle=True, verbose=1) 
    """datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    
    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train), epochs=epochs, validation_data=datagen.flow(x_val, y_val, batch_size=batch_size), validation_steps=x_val.shape[0])"""
    return model, hist.history


def eval_plot(model, x_eval, y_eval, history, epochs):
    

    scores = model.evaluate(x_eval, y_eval, batch_size = 128,verbose=0)
    print(scores)
    
    
    epochs_array = np.arange(epochs)
    plt.figure(1)
    plt.title('Loss in %d epochs' %(epochs))
    plt.plot(epochs_array, np.asarray(history['loss']))
    plt.legend(['loss'])
    
    plt.figure(2)
    plt.title('Accuracy in %d epochs' %(epochs))
    plt.plot(epochs_array, np.asarray(history['acc']))
    plt.legend(['acc'])
    
    plt.figure(3)
    plt.title('Val_Loss in %d epochs' %(epochs))
    plt.plot(epochs_array, np.asarray(history['val_loss']))
    plt.legend(['val_loss'])

    plt.figure(4)
    plt.title('Val_Accuracy in %d epochs' %(epochs))
    plt.plot(epochs_array, np.asarray(history['val_acc']))
    plt.legend(['val_acc'])


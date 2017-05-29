import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.optimizers import SGD



def sgdOpt(learningrate= 0.01, momentum= 0.0,  decay= 0.0, nestrov=False):
    
    opt = SGD(lr= learningrate, momentum=momentum, decay=decay, nesterov=nestrov)
    
    return opt
    

def rmsPropOpt(learningrate=0.00025, rho=0.9, epsilon=1e-08, decay=0.0):
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr = learningrate, rho=rho, epsilon= epsilon, decay = decay)
        
    return opt


def adagradOpt(learningrate= 0.01, epsilon= 1e-08,  decay= 0.0):
    
    opt = keras.optimizers.Adagrad(lr= learningrate, epsilon= epsilon, decay= decay)

    return opt

def adamOpt(learningrate= 0.001, epsilon= 1e-08,  decay= 0.0):

    opt = keras.optimizers.Adam(lr= learningrate, beta_1=0.9, beta_2=0.999, epsilon= epsilon, decay= decay)
    
    return opt

def compiling(model, opt):
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model
    
    
def training(model, batch_size, epochs, x_train, y_train):

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
              validation_split=0.3, shuffle=True, verbose=1)
    
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


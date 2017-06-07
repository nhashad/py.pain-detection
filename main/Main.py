import utils.data_prc as dp 
import utils.build_network as bn
import utils.compilation_opt as cpo
%pylab inline


np.random.seed(42)
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from datetime import datetime
import time
import argparse

# reset everything to rerun in jupyter
tf.reset_default_graph()

batch_size = 128
num_classes = 7
epochs = 20

layer1_size = 32

def train_model(train_file='fer2013.pickle', job_dir='./tmp/main-1', **args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_file))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')
    #file_stream = file_io.FileIO(train_file, mode='r')
    x_train, y_train, x_test, y_test  = dp.dataset_loading_emotion(train_file.split('.')[0])
    
    #x_train = x_train.toarray()
    #x_test = x_test.toarray()
    
    x_train /= np.max(x_train)
    x_test /= np.max(x_test)

    print(x_train.shape, y_train.shape, 'train samples,', type(x_train[0][0]), ' ', type(y_train[0][0]))
    print(x_test.shape,  y_test.shape,  'test samples,',  type(x_test[0][0]),  ' ', type(y_train[0][0]))

    # convert class vectors to binary class matrices. Our input already made this. No need to do it again
    y_train, y_test = dp.y_to_categorical( y_train, y_test, num_classes)

    model = bn.build_model(x_train)
    opt = cpo.adamOpt() 
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer= opt,
                  metrics=['accuracy'])
    model, history = cpo.training(model, batch_size, epochs, x_train, y_train)

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
from __future__ import division
import numpy as np
from numpy import matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from keras.datasets import mnist
from keras.callbacks import TensorBoard

np.random.seed(1000)

# ------------------Helper Functions----------------------- #

def create_MNIST_dataset_feedforward():
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype(float)

    # reshape label vector so it has a second dimension of 1
    y = np.reshape(mnist.target, (mnist.target.shape[0], 1))

    # split up data set into training and test sets and shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # scale training and test data to unit norm
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # these don't have an extra column of 1s
    X_train_keras = X_train
    X_test_keras = X_test

    # put in column of 1s for bias variable
    X_train = matrix(np.hstack((np.ones((X_train.shape[0], 1)), X_train)))
    X_test = matrix(np.hstack((np.ones((X_test.shape[0], 1)), X_test)))

    """
    # test some images to make sure they are labeled correctly
    from PIL import Image

    w, h = 28, 28
    for i in [12,234,121,66,788,3444]:
        print X_train_keras[0,:].shape
        data = np.reshape(X_train_keras[i,:],(28,28))
        print "LABEL", y_train[i]
        img = Image.fromarray(data, 'L')
        img.show()
        stop = raw_input("enter something to continue")
    """

    # convert labels to binary array
    y_train_keras = np_utils.to_categorical(y_train)
    y_test_keras = np_utils.to_categorical(y_test)

    # print X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_train_keras.shape, X_test_keras.shape
    return X_train_keras, X_test_keras, y_train_keras, y_test_keras

def create_MNIST_dataset_convolutional():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # scale training and test data to unit norm
    X_train = np.reshape(X_train, (X_train.shape[0], 784))
    X_test = np.reshape(X_test, (X_test.shape[0], 784))
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    # these don't have an extra column of 1s
    X_train_keras = X_train
    X_test_keras = X_test

    """
    # test some images to make sure they are labeled correctly
    from PIL import Image

    w, h = 28, 28
    for i in [12,234,121,66,788,3444]:
        print X_train_keras[0,:].shape
        data = np.reshape(X_train_keras[i,:],(28,28))
        print "LABEL", y_train[i]
        img = Image.fromarray(data, 'L')
        img.show()
        stop = raw_input("enter something to continue")
    """

    # convert labels to binary array
    y_train_keras = np_utils.to_categorical(y_train)
    y_test_keras = np_utils.to_categorical(y_test)

    #print X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_train_keras.shape, X_test_keras.shape
    return X_train_keras, X_test_keras, y_train_keras, y_test_keras

# ------------------Feed Forward Network----------------------- #
# use "tensorboard --logdir /tmp/tf_board/feedforward" in terminal to access TensorBoard

# get data
X_train_FFN, X_test_FFN, y_train_FFN, y_test_FFN = create_MNIST_dataset_feedforward()

# set hyperparameters and dimensions
batch_size = 30
input_dim = 784
learning_rate = 0.05
epochs = 5

# do train a feed forward network on MNIST
model = Sequential()
model.add(Dense(500, input_dim=input_dim, activation='relu'))
model.add(Dense(500, input_dim=5, activation='relu'))
model.add(Dense(500, input_dim=5, activation='relu'))
model.add(Dense(10, input_dim=5, activation='softmax'))

sgd = SGD(lr=learning_rate, decay=1e-6)
tb_callback = TensorBoard(log_dir='/tmp/tf_board/feedforward')

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train_FFN, y_train_FFN, nb_epoch=epochs, batch_size=batch_size, callbacks=[tb_callback])

trained_FFN = model

# ------------------Convolutional Network----------------------- #
# use "tensorboard --logdir /tmp/tf_board/convolutional" in terminal to access TensorBoard
# get data
X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = create_MNIST_dataset_convolutional()

# set hyperparameters and dimensions
batch_size = 30
input_shape = (28, 28, 1)
learning_rate = 0.05
epochs = 5

# train a convolutional neural network
model = Sequential()
model.add(Convolution2D(60, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))
model.add(Convolution2D(60, 5, 5, border_mode='valid', input_shape=model.layers[-1].output_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Flatten())
model.add(Dense(300, input_shape=model.layers[-1].output_shape, activation='relu'))
model.add(Dense(150, input_shape=model.layers[-1].output_shape, activation='relu'))
model.add(Dense(10, input_shape=model.layers[-1].output_shape, activation='softmax'))

sgd = SGD(lr=learning_rate, decay=1e-6)
tb_callback = TensorBoard(log_dir='/tmp/tf_board/convolutional')

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train_CNN, y_train_CNN, nb_epoch=epochs, batch_size=batch_size, callbacks=[tb_callback])

trained_CNN = model

# ------------------Print Output----------------------- #

print ""
print "***** FEED FORWARD NET, MINST *****"
print ""
scores = trained_FFN.evaluate(X_train_FFN, y_train_FFN, verbose=0)
for metric, value in zip(scores, trained_FFN.metrics_names):
    print "TRAINING", metric, value
print ""
scores = trained_FFN.evaluate(X_test_FFN, y_test_FFN, verbose=0)
for metric, value in zip(scores, trained_FFN.metrics_names):
    print "TESTING", metric, value

print ""
print "***** CONVOLUTIONAL NET, MNIST *****"
print ""
scores = trained_CNN.evaluate(X_train_CNN, y_train_CNN, verbose=0)
for metric, value in zip(scores, trained_CNN.metrics_names):
    print "TRAINING", metric, value
print ""
scores = trained_CNN.evaluate(X_test_CNN, y_test_CNN, verbose=0)
for metric, value in zip(scores, trained_CNN.metrics_names):
    print "TESTING", metric, value



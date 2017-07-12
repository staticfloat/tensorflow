
# coding: utf-8

# In[2]:


import tensorflow as tf
import keras
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Model
from keras.engine.topology import Layer


class PhillyLogger(keras.callbacks.Callback):
    def __init__(self, logdir, total_epochs, total_batches, **kwargs):
        self.logdir=logdir
        self.epochs=total_epochs
        self.minibatches = total_batches
        super(PhillyLogger, self).__init__(**kwargs)

    def on_train_begin(self, logs={}):
        self.logger = phillylogger(self.logdir, total_epoch=self.epochs, total_minibatch=self.minibatches)

    def on_batch_end(self, batch, logs={}):
        self.logger.minibatch_complete(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        self.logger.epoch_complete(logs.get('loss'))

    def on_train_end(self, logs={}):
        self.logger.logging_complete()

# In[3]:


class TestLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TestLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(TestLayer, self).build(input_shape)
    def call(self, x):
        return tf.nn.dropout(x, 0.5)
    def compute_output_shape(self, input_shape):
        return input_shape


# In[4]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[6]:


X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)


# In[7]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[8]:


# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# In[9]:


inp = Input(shape=(32,32,3))
z = Convolution2D(32, (3,3), activation='relu')(inp)
z = Convolution2D(32, (3,3), activation='relu')(z)
z = MaxPooling2D(pool_size=(2,2))(z)
z = Dropout(0.25)(z)

z = Flatten()(z)
z = Dense(128, activation='relu')(z)
#z = Dropout(0.5)(z)
z = TestLayer(0.5)(z)
z = Dense(10, activation='softmax')(z)

model = Model(inputs=inp, outputs=z)

"""model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))"""


# In[10]:
callback = []
philly_logger = PhillyLogger(logdir, epochs, epochs*(train_gen.samples / train_gen.batch_size))
callbacks.append(philly_logger)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[23]:


model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1, callbacks=callbacks)


# In[21]:


score = model.evaluate(X_test, Y_test, verbose=0)


# In[22]:


score
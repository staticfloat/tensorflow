
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, AveragePooling2D, Reshape, Activation
from keras.layers.advanced_activations import PReLU
from keras import backend as K
from math import sqrt
import sys
from TB_writer import TB_writer
sys.path.append('../CustomLayers/')
from CustomLayers import *


# In[2]:


train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[3]:


train_generator = train_datagen.flow_from_directory(
        '/data/cifar/train/',
        target_size=(32,32),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '/data/cifar/test/',
        target_size=(32,32),
        batch_size=32,
        class_mode='categorical')


# In[10]:


img_w = 32
img_h = 32
img_c = 3
inp = Input(shape=(img_w, img_h, img_c))

z = Convolution2D(32, (3,3), activation='relu')(inp)
z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)
z = BatchNormalization()(z)
#z = Convolution2D(32, (3,3), activation='relu')(z)
z = BinLayer()(z)
#z = MultibitLayer(2)(z)
#z = BinConv(128, (3,3), kernel_regularizer=BinReg(), padding='same')(z)
z = Convolution2D(128, (3,3), activation='relu', padding='same')(z)
z = PReLU()(z)
z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)

z = BatchNormalization()(z)
z = BinLayer()(z)
#z = MultibitLayer(2)(z)
#z = BinConv(128, (3,3), kernel_regularizer=BinReg(), padding='same')(z)
z = Convolution2D(128, (3,3), activation='relu', padding='same')(z)
z = PReLU()(z)
z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)

z = BatchNormalization()(z)
z = Convolution2D(10, (1,1), activation='relu')(z)
z = AveragePooling2D(pool_size=(int(z.shape[1]), int(z.shape[2])))(z)
z = Reshape((10,))(z)
z = Activation('softmax')(z)

model = Model(inputs=inp, outputs=z)


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


tb_callback = TB_writer(histogram_freq=1, write_images=True, log_dir="cifar_test_binary", val_gen=validation_generator)
tb_callback.set_model(model)


# In[ ]:


model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=100,
        callbacks=[tb_callback])


# In[ ]:


from PIL import Image


# In[ ]:


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float32" )
    data = data/255
    return data


# In[ ]:


image = load_image("/data/cifar/test/truck/1008_truck.png")
image = image.reshape((1,)+image.shape)


# In[ ]:


class_map =validation_generator.class_indices


# In[ ]:


guess = np.argmax(model.predict(image))


# In[ ]:


validation_generator.class_indices


# In[ ]:


cifar_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[ ]:


cifar_labels[guess]


# In[ ]:





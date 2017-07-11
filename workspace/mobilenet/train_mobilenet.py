
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import numpy as np
from mobilenet import MobileNet, BinMobileNet
import sys
sys.path.append('../CustomLayers/')
sys.path.append('../Keras/')
from TB_writer import TB_writer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input


# In[ ]:


get_ipython().magic(u'env CUDA_VISIBLE_DEVICES=0')


# In[2]:


train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[3]:


train_generator = train_datagen.flow_from_directory(
        '/data/imagenet/train/',
        target_size = (224, 224),
        batch_size=128,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '/data/imagenet/val/',
        target_size = (224, 224),
        batch_size=128,
        class_mode='categorical')


# In[4]:


img_w = 224
img_h = 224
img_c = 3

inp = Input(shape=(img_w, img_h, img_c))

model = MobileNet(input_tensor=inp, shallow=True)


# In[5]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[6]:


model.summary()


# In[7]:


tb_callback = TB_writer(histogram_freq=5, write_images=True, log_dir="/data/tensorflow/log/binmobilenet", val_gen=validation_generator)
tb_callback.set_model(model)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=3, min_lr=0.0001)

checkpoint_callback = keras.callbacks.ModelCheckpoint("/data/tensorflow/log/binmobilenet" + ".{epoch:02d}-{val_loss:.2f}.hdf5")


# In[8]:


model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples / train_generator.batch_size,
        epochs = 10,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples / validation_generator.batch_size,
        callbacks=[tb_callback, reduce_lr])


# In[10]:


model.save("binmobilenet.hdf5")


# In[ ]:





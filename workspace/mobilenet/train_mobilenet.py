import keras
import tensorflow as tf
import numpy as np
import mobilenet
import sys
import argparse
sys.path.append('../CustomLayers/')
sys.path.append('../Keras/')
from TB_writer import TB_writer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from phillylogger import *

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

def setup_data(datadir, minibatch_size):
# set up data reader

    train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # point data to the right place

    train_generator = train_datagen.flow_from_directory(
            datadir + '/train',
            target_size = (224, 224),
            batch_size=minibatch_size,
            class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
            datadir + '/val',
            target_size = (224, 224),
            batch_size=minibatch_size,
            class_mode='categorical')

    return train_generator, validation_generator


def mobilenet_train(train_gen, val_gen, epochs=10, restore=False, logdir=None, model_name=None, model_path=None):

    # basic input shape and model creation

    img_w = 224
    img_h = 224
    img_c = 3

    inp = Input(shape=(img_w, img_h, img_c))

    model_func = getattr(mobilenet, model_name)
    model = model_func(inp)

    # see if the weights can be loaded
    if (restore == True and model_path != None):
        try:
            model.load_weights(model_path + "/" + model_name)
        except:
            pass

    # build the model

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # set up callbacks
    callbacks = []

    if logdir != None:
        # tensorboard instantiation
        tb_callback = TB_writer(histogram_freq=5, write_images=True, log_dir=logdir+ "/" + model_name, val_gen=val_gen)
        tb_callback.set_model(model)
        callbacks.append(tb_callback)
    
        #philly logger instantiation
        philly_logger = PhillyLogger(logdir, epochs, epochs*(train_gen.samples / train_gen.batch_size))
        callbacks.append(philly_logger)


    if model_path != None:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(model_path + "/" + model_name + ".{epoch:02d}-{val_loss:.2f}.hdf5")
        callbacks.append(checkpoint_callback)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=5, min_lr=0.0001)
    callbacks.append(reduce_lr)
    # do the training

    model.fit_generator(
            train_gen,
            steps_per_epoch = train_gen.samples / train_gen.batch_size,
            epochs = epochs,
            validation_data = val_gen,
            validation_steps = val_gen.samples / val_gen.batch_size,
            callbacks=callbacks,
            verbose=0)

    # save the model
    if model_path != None:
        model.save(model_path + "/" + model_name +".hdf5")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    data_path = "/data/imagenet/"
    
    parser.add_argument('-datadir', '--datadir', help='Data directory where the imagenet dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='10')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default=128)
    parser.add_argument('-r', '--restart', help="Whether to restart from scratch", action='store_true')
    parser.add_argument('-model', '--model', type=str, help="which model to train with", required=True, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['logdir'] is not None:
        log_dir = args['logdir']
    
    data_path = args['datadir']
    
    if not os.path.isdir(data_path):
        raise RunTimeError("Directory %s does not exist" % data_path)
    
    train_gen, val_gen = setup_data(data_path, args['minibatch_size'])

    mobilenet_train(train_gen, val_gen,
                    epochs=args['num_epochs'],
                    restore=args['restart'],
                    logdir=args['logdir'],
                    model_name=args['model'],
                    model_path=args['outputdir'])

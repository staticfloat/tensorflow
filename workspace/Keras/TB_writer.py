import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras import backend as K
from math import sqrt

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x

class TB_writer(keras.callbacks.Callback):
    def __init__(self, log_dir="",
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=True,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 val_gen=None):
        super(TB_writer, self).__init__()
        self.log_dir = "/data/tensorflow/log/"+log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.batch_size = batch_size
        self.merged = None
        self.val_gen = val_gen
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    if len(weight.shape) == 4:
                        kernel_split = tf.split(weight, weight.shape[3], axis=3)
                        i = 0
                        for kernel in kernel_split:
                            tf.summary.histogram(mapped_weight_name + str(i), kernel)
                            i += 1
                    else:
                        tf.summary.histogram(mapped_weight_name, weight)
                    
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss, weight)
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape)==2: #dense layer
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1, shape[0], shape[1], 1])
                            w_img = tf.transpose(w_img)
                        elif len(shape) == 4: #convnet check
                            w_img = put_kernels_on_grid(w_img)
                            #if K.image_data_format() == 'channels_last':
                            #    #w_img = tf.transpose(w_img, perm[2, 0, 1])
                            #    w_img = tf.transpose(w_img, perm=[3, 2, 0, 1])
                            #    shape = K.int_shape(w_img)
                            # break kernel into black and white per channel
                            #imgs = tf.split(w_img)                            
                            #w_img = tf.reshape(w_img [shape[0], shape[1], shape[2], 1])
                            w_img = tf.transpose(w_img, perm=[3, 1, 2, 0])
                        elif len(shape)==1: #bias case
                            w_img = tf.reshape(w_img, [1, shape[0], 1, 1])
                            w_img = tf.transpose(w_img)
                        else:
                            # maybe cant handle 3d convnnets
                            continue
                        shape = K.int_shape(w_img)
                        #print(shape)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name,w_img, max_outputs=8)
                        
                if hasattr(layer, 'output'):              
                    mapped_layer_name = layer.name.replace(':', '_')
                    if len(layer.output.shape) == 4:                        
                        output_split = tf.split(layer.output, layer.output.shape[3], axis=3)
                        i = 0
                        for output in output_split:
                            tf.summary.histogram('{}/out'.format(mapped_layer_name) + str(i), output)
                            tf.summary.image('{}/out'.format(mapped_layer_name) + str(i), output)
                            i += 1
                            if i > 16:
                                break;
                    else:
                        tf.summary.histogram('{}/out'.format(mapped_layer_name), layer.output)                        
                    
            self.merged = tf.summary.merge_all()
            if self.write_graph:
                self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir)
                
            if self.embeddings_freq:
                embeddings_layer_names = self.embeddings_layer_names

                if not embeddings_layer_names:
                    embeddings_layer_names = [layer.name for layer in self.model.layers
                                              if type(layer).__name__ == 'Embedding']

                embeddings = {layer.name: layer.weights[0]
                              for layer in self.model.layers
                              if layer.name in embeddings_layer_names}

                self.saver = tf.train.Saver(list(embeddings.values()))

                embeddings_metadata = {}

                if not isinstance(self.embeddings_metadata, str):
                    embeddings_metadata = self.embeddings_metadata
                else:
                    embeddings_metadata = {layer_name: self.embeddings_metadata
                                           for layer_name in embeddings.keys()}

                config = projector.ProjectorConfig()
                self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                         'keras_embedding.ckpt')

                for layer_name, tensor in embeddings.items():
                    embedding = config.embeddings.add()
                    embedding.tensor_name = tensor.name

                    if layer_name in embeddings_metadata:
                        embedding.metadata_path = embeddings_metadata[layer_name]

                projector.visualize_embeddings(self.writer, config)
                
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.val_gen and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                val_data = self.val_gen.next() + ([1], )
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
                
                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]
                    val_data += ((True, ))                  

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size
                    
        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)
                
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()  
        
def write_TB(model, log_dir, data_gen):
    proxy_callback = TB_writer(histogram_freq=1, write_images=True, log_dir=log_dir, val_gen=data_gen)
    proxy_callback.set_model(model)
    proxy_callback.on_epoch_end(epoch=1)
    proxy_callback.on_train_end(0)
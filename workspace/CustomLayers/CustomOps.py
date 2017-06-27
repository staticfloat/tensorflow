import tensorflow as tf
from tensorflow.python.framework import ops

binarize_module = tf.load_op_library("binarize.so")
multibit_module = tf.load_op_library("multibit.so")

@ops.RegisterGradient("Binarize")
def bin_grad(op, grad):
    in_vals = op.inputs[0]
    grad_func = tf.cast(tf.less_equal(tf.abs(in_vals), 1), tf.float32)
    grad_out = tf.multiply(grad_func, grad)
    return [grad_out]

@ops.RegisterGradient("Multibit")
def multibit_grad(op, grad):
    in_vals = op.inputs[0]
    grad_func = tf.cast(tf.less_equal(tf.abs(in_vals), 1), tf.float32)
    grad_out = tf.multiply(grad_func, grad)
    return [grad_out, tf.zeros(tf.shape(op.inputs[1]))]

def binarize(x):
    return binarize_module.binarize(x)

def multibit(x, bits):
    return multibit_module.multibit(x, bits)

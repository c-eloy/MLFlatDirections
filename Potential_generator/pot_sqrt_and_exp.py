
import tensorflow as tf
import numpy as np

dim = 2

def V(x):
    """
	V = tf.exp((-1 + x1 ** 2 + x2 ** 2) ** 2)*tf.sqrt(x1**2+x2**2)
    """

    x1, x2 = tf.split(x, 2, axis=1)

    return tf.exp((-1 + x1 ** 2 + x2 ** 2) ** 2)*tf.sqrt(x1**2+x2**2)

import tensorflow as tf
import numpy as np

dim = 3

def V(x):
    """
    0.5 * (4 + tf.exp(6 * x1) + tf.exp(-2 * x1 -4 * (3 ** -0.5) * x2) -2 * (tf.exp(2 * x1 -2 * (3 ** -0.5) * x2)) -4 * (tf.exp(-2 * x1 -((3 ** -0.5) * x2))) -4 * (tf.exp(2 * x1 + (3 ** -0.5) * x2)) -4 * (x3 ** 2) + 2 * (tf.exp(-4 * x1 -2 * (3 ** -0.5) * x2)) * (x3 ** 2) + 2 * (tf.exp(4 * x1 + 2 * (3 ** -0.5) * x2)) * (x3 ** 2) + (tf.exp(-6 * x1)) * (x3 ** 4) -2 * (tf.exp(-2 * x1 + 2 * (3 ** -0.5) * x2)) * (x3 ** 4) + (tf.exp(2 * x1 + 4 * (3 ** -0.5) * x2)) * (x3 ** 4))
    """

    x1,x2,x3=tf.split(x, 3, axis=1)

    return 0.5 * (4 + tf.exp(6 * x1) + tf.exp(-2 * x1 -4 * (3 ** -0.5) * x2) -2 * (tf.exp(2 * x1 -2 * (3 ** -0.5) * x2)) -4 * (tf.exp(-2 * x1 -((3 ** -0.5) * x2))) -4 * (tf.exp(2 * x1 + (3 ** -0.5) * x2)) -4 * (x3 ** 2) + 2 * (tf.exp(-4 * x1 -2 * (3 ** -0.5) * x2)) * (x3 ** 2) + 2 * (tf.exp(4 * x1 + 2 * (3 ** -0.5) * x2)) * (x3 ** 2) + (tf.exp(-6 * x1)) * (x3 ** 4) -2 * (tf.exp(-2 * x1 + 2 * (3 ** -0.5) * x2)) * (x3 ** 4) + (tf.exp(2 * x1 + 4 * (3 ** -0.5) * x2)) * (x3 ** 4))
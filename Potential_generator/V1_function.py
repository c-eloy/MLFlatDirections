import tensorflow as tf
import numpy as np

def V(x):
    """
    V = x2**2 - 2*x2*x3 + x3**2 + 2*x2*x4 - 2*x3*x4 + 2*x4**2
    """
    x1,x2,x3,x4=tf.split(x, 4, axis=1)

    expr = x2**2 - 2*x2*x3 + x3**2 + 2*x2*x4 - 2*x3*x4 + 2*x4**2

    return expr

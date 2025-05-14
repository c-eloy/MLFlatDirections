import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import keras
from keras import layers,models,regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-latent_dim", help="", default = 1, type = int)
parser.add_argument("-input_dim", help="", default = 1, type = int)
parser.add_argument("-n_epochs", help="", default = 1, type = int)
parser.add_argument("-batch_size", help="", default = 1, type = int)
parser.add_argument("-n_neurons", help="", default = 1, type = int)
parser.add_argument("-path", help="", default = 'None', type = str)

parser.parse_args()
args = parser.parse_args()

this_latent_dim = args.latent_dim
input_dim = args.input_dim
n_neurons = args.n_neurons

x = np.load(args.path)
x = tf.Variable(x)

set_train, set_test = train_test_split(x.numpy(), test_size=0.1, random_state=1)

inputs = layers.Input(shape=(input_dim,))
encoded = layers.Dense(n_neurons, activation='sigmoid')(inputs)
encoded = layers.Dense(n_neurons, activation='sigmoid')(encoded)
encoded = layers.Dense(this_latent_dim)(encoded)  

decoded = layers.Dense(n_neurons, activation='sigmoid')(encoded)
decoded = layers.Dense(n_neurons, activation='sigmoid')(decoded)
decoded = layers.Dense(input_dim)(decoded)  

this_autoencoder = keras.Model(inputs, decoded)
this_autoencoder.compile(optimizer='adam', loss='mse')

history = this_autoencoder.fit(set_train, set_train, epochs=args.n_epochs, batch_size=args.batch_size, shuffle=True, validation_data=(set_test,set_test))
    
np.save(f"History_{input_dim}_{args.n_epochs}_{args.n_neurons}",history)
this_autoencoder.save(f'ae_{input_dim}_{args.n_epochs}_{args.n_neurons}.keras')

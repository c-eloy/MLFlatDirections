#import keras
#from keras import layers,models,regularizers
#import tensorflow as tf
import argparse
import reconstruct_pol_with_NN_module as mod
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-n_neurons", help="Number of neurons for the layers", default = 256, type = int)
parser.add_argument("-max_order", help="Maximum power to look for", default = 4, type = int)
parser.add_argument("-n_epochs_for_orders", help="Number of epochs when looking for the powers", default = 50, type = int)
parser.add_argument("-n_epochs_for_model", help="Number of epochs when learning one power", default = 300)
parser.add_argument("-isolate", help="Which x to isolate for the analysis", type = int)
parser.add_argument("-save", help="Wether or not to save result", default = True)
parser.add_argument("-tol_order", help="Tolerance when choosing the orders", default = 10**(-1))
parser.add_argument("-file", help="File where to save the data", type = str, default = "")
parser.parse_args()
args = parser.parse_args()
#print(args.echo)

all_X = np.load("all_X_10dpot_scr.npy")

mod.get_NN_one_x(all_X = all_X, 
                 isolate = args.isolate,
                 max_order = args.max_order,
                 n_epochs_for_orders = args.n_epochs_for_orders,
                 n_epochs_for_model = args.n_epochs_for_model,
                 tol_order = args.tol_order,
                 save = args.save,
                 n_neurons = args.n_neurons,
                 file = args.file)


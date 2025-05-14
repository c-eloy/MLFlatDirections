import numpy as np 
import tensorflow as tf 
import sys
import os
import time 
#module_path = "/home/bbastien/MLFlatDirections"
module_path = "/Users/bastienduboeuf/Documents/Recherche/MLForFlatDirections3D/MLFlatDirections"
sys.path.append(module_path)

import grad_descent_module as gdm 
import potentials as pot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-file_of_points", help="File where we should take the points from. If 'None', randomly initialize the points", default = "None", type = str)
parser.add_argument("-out_file_of_points", help="File where we should save the points.", default = "None", type = str)
parser.add_argument("-n_points", help="Numbers of point to initialize", default = 10000, type = int)
parser.add_argument("-update_opt", help="Wether to reinitialize the optimizer", default = False, type = bool)
parser.add_argument("-n_steps", help="Number of epochs for the gradient descent", default = 1000, type = int)
parser.add_argument("-n_steps_opt", help="After how many steps should the optimier be reinitialize", default = 200, type = int)
parser.add_argument("-step_stop_opt", help="After how many steps should the reinitialization be stopped", default = 1500, type = int)
parser.add_argument("-alpha", help="Learning rate", default = 10**(-2), type = float)
parser.add_argument("-n_step_print", help="Frequency for printing the output", default = 10, type = int)
parser.add_argument("-n_var", help="Number of variable", default = 13, type = int)
#parser.add_argument("-pot", help="Name of the potential", default =, type = int)

parser.parse_args()
args = parser.parse_args()

n_points = args.n_points
if args.file_of_points != 'None':
    x = tf.Variable(np.load(args.file_of_points))
else: 
    this_seed = int((time.time()-int(time.time()))*1000)
    np.random.seed(this_seed)
    x = tf.Variable(2*np.random.rand(n_points,args.n_var)-1, dtype=tf.float32)

this_history,this_x = gdm.grad_descent_potential(pot.V3d_sugra, 
                                                 x, 
                                                 update_opt = args.update_opt, 
                                                 n_steps = args.n_steps,
                                                 n_steps_opt = args.n_steps_opt, 
                                                 step_stop_opt = args.step_stop_opt, 
                                                 alpha = args.alpha,
                                                 n_step_print = args.n_step_print, 
                                                 thres = 10)

if args.file_of_points != "None":
    np.save(args.file_of_points,this_x)
    np.save(args.file_of_points+"_hist",this_history)
else: 
    np.save("Points_"+args.out_file_of_points+'_'+time.strftime("%H_%M_%S")+'_seed'+str(this_seed),this_x)
    np.save("Loss_"+args.out_file_of_points+'_'+time.strftime("%H_%M_%S")+'_seed'+str(this_seed),this_history)

for i in range(15):
    print(f"Number of points with ||∇V||^2 > {10**(-i)} : {sum(tf.reduce_sum(gdm.get_grad(pot.V3d_sugra,this_x)**2,axis=1).numpy()>10**(-i))}")



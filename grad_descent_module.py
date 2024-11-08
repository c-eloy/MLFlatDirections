import keras
from keras import layers,models,regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
import tensorflow as tf
import numpy as np

# Oprimizer for the gradient descent

# compute (||∇V||^2)
def grad_norm_squared(V, x ,factor=1):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = V(x)  # compute V at each point
    gradients = tape.gradient(loss, x)  # compute ∇V at each point
    norm_squared = tf.reduce_sum(gradients**2, axis=1)  # ||∇V||^2 at each point
    return norm_squared * factor

def get_grad(V, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = V(x)  # compute V at each point
    gradients = tape.gradient(loss, x)  # compute ∇V at each point
    return gradients

def grad_descent_potential(V, x, history_loss = [], update_opt = True, n_steps = 1500, n_steps_opt = 200, step_stop_opt = 1200, alpha=10**(-3), n_step_print = 10, thres = 10):

    """ Does the gradien descent for the point x in the potential V. 
        - V : potential to analyse, function
        - x : intial point on which we do the gradient descent, tensorflow Variable (Tensor) of shape (n_points,n_sample)
        - update_opt : whether or not we reinitialze the optimizer after n_steps in the gradient descent, bool. 
        - n_steps : number of steps for the gradient descent, int. 
        - n_steps_opt : if update_opt is True, rate at which we reinitialze the optimizer, int.
        - step_stop_opt : number of steps after which we stop the reinitialization of the optimizer, int. 
        - alpha : learning_rate, float. 
        - n_step_print : rate at which we print the value of the loss, int. 
        - thres : if loss is smaller than 10**(-thres), stop the gradient descent, float.
    """

    optimizer = tf.optimizers.Adam(learning_rate=alpha)
    
    loss_prev_step=10**8

    for step in range(n_steps):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(grad_norm_squared(V, x)) # minimize ||∇V||^2 for all points
    
        # Compute ||∇V||^2 with respect to x
        gradients = tape.gradient(loss, [x])
    
        if update_opt and step % n_steps_opt ==0 :
            print("Reinitializing the optimizer")
            optimizer = tf.optimizers.Adam(learning_rate=alpha)
    
        if step == step_stop_opt:
            update_opt = False
        
        if step == 2500:
            optimizer = tf.optimizers.Adam(learning_rate=0.001)
        if step == 5000:
            optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        if step == 7500:
            optimizer = tf.optimizers.Adam(learning_rate=0.00001)

            #optimizer.learning_rate.assign(0.0001)

        # Apply gradient descent for all points
        optimizer.apply_gradients(zip(gradients, [x]))

        history_loss.append(loss)
    
        if step % n_step_print == 0:
            print(f"""Step {step}: ||∇V||^2 = {loss.numpy()} with learning rate {optimizer.get_config()["learning_rate"]}""")
    
        if np.log(loss.numpy())/np.log(10)<-thres:
            print("Converged enough")
            print(f"Step {step}: ||∇V||^2 = {loss.numpy()}")
            break
    return history_loss,x    


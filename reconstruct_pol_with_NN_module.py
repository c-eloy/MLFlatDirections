import keras
from keras import layers,models,regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def find_order(X, Y, max_order = 4, n_epochs = 50, n_neurons = 128, input_shape = 9, batch_size = 100, lambda_reg = 0):

    """Function that seeks to what order a given coordinate probably appears in a polynomial. Learns Y**order in
    terms of X, for orders in range(max_order). Then returns the losses, from which we can extract the probable
    best orders.
        - X : x_data for the neural network. Should contain different x's which you are going to use to reconstruct
            Y. array of shape (n_var,n_samples)
        - Y : y_data. The variable we try to reconstruct with x's. array of shape (n_samples,)
        - max_order. Maximum order to analyse. int
        - n_epochs. Numbers of epoch to train the NN with. int
        - n_neurons. The architecture is fixed to dense layers of shape (n_var,n_neurons,n_neurons,1). int
        - input_shape. Number of variables n_var. int
        - batch_size. batch_size for the training of the NN. int
    """

    all_loss = []
    for i in range(1,max_order+1):
        this_model = tf.keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(n_neurons, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_reg), bias_regularizer=regularizers.l2(lambda_reg)),
            layers.Dense(n_neurons, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_reg), bias_regularizer=regularizers.l2(lambda_reg)),
            layers.Dense(1, kernel_regularizer=regularizers.l2(lambda_reg), bias_regularizer=regularizers.l2(lambda_reg))
])
        this_model.compile(optimizer='adam', loss='mse')
        hist_this_loss = this_model.fit(X, Y**i, epochs=n_epochs, batch_size=batch_size)
        all_loss.append([i,hist_this_loss.history["loss"]])
    return all_loss

def learn_one_x_from_the_others(X, Y, n_neurons = 256, order = 1, lambda_reg = 0, n_epochs = 200, batch_size = 100, factor_red_lr = 0.5):

    """Train a NN to learn Y**order in terms of X. The architecture is fixed to dense layers of
    shape (n_var,n_neurons,n_neurons,1).
        - X : x_data for the neural network. Should contain different x's which you are going to use to reconstruct
            Y. array of shape (n_var,n_samples)
        - Y : y_data. The variable we try to reconstruct with x's. array of shape (n_samples,)
        - order : power to raise Y. int
        - lambda_reg : regularisation parameter for the NN. float
        - n_epochs : number of epochs for the training. int
        - batch_size : batch_size for the training. int
    """

    input_shape = 9
    this_model = tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(n_neurons, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_reg), bias_regularizer=regularizers.l2(lambda_reg)),
        layers.Dense(n_neurons, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_reg), bias_regularizer=regularizers.l2(lambda_reg)),
        layers.Dense(1, kernel_regularizer=regularizers.l2(lambda_reg), bias_regularizer=regularizers.l2(lambda_reg))
        ])

    this_model.compile(optimizer=keras.optimizers.Adam(learning_rate=10**(-2)), loss='mse')

    X_train, X_test = train_test_split(X, test_size=0.1, random_state=1)
    Y_train, Y_test = train_test_split(Y, test_size=0.1, random_state=1)

    if factor_red_lr < 1:
        #reduce_lr = ReduceLROnPlateau(monitor='loss',
        #                              factor=factor_red_lr,
        #                              patience=10,
        #                              min_lr=10**(-6),
        #                              min_delta = 10**(-3))
        def scheduler(epoch, lr):
            if epoch < 100:
                return 10**(-2)
            elif epoch < 150:
                return 10**(-3)
            elif epoch < 200: 
                return 10**(-4)
            else: 
                return 10**(-5)

    
        lr_schedule = LearningRateScheduler(scheduler)
        this_hist_loss = this_model.fit(X, Y**order, epochs=n_epochs, batch_size=batch_size, callbacks=[lr_schedule], validation_data=(X_test,Y_test))
    else:
        this_hist_loss = this_model.fit(X, Y**order, epochs=n_epochs, batch_size=batch_size, validation_data=(X_test,Y_test))

    return [this_model,this_hist_loss]

def save_nn(model_to_save, name, file):

    """Save the NN
        - model_to_save : which NN to save.
        - name : name under which to save NN. str
    """

    index = 0
    for layer in model_to_save.layers:
        this_weights, this_baises = layer.get_weights()
        np.savetxt(str(file)+"/w"+str(index)+"_model_"+name+".csv", this_weights, delimiter=",",fmt="%.6f")
        np.savetxt(str(file)+"/b"+str(index)+"_model_"+name+".csv", this_baises, delimiter=",",fmt="%.6f")
        index += 1

def select_order(list_losses, tol = 0.1):

    """Find the power for a given X, Y to which we should look a polynomial : Y**order = P(X). Look for minima in
    loss histories.
        - list_losses : list of losses of a NN trying to learn Y**order with X, previously obtained
        - tol : select which order got smallest loss, and retain all minima with tolerance in % tol
    """

    index_min = 0
    best_min = 10**8
    list_min = []
    to_explore = []
    for i in list_losses:
        index_min = 0
        this_min = np.min(i[1])
        list_min.append([i[0],this_min])
        if this_min < best_min:
            best_min  = this_min
            index_min = i[0]
    to_explore.append(index_min)

    for i in range(len(list_losses)):
        if i == index_min:
            continue

        elif list_min[i][1] * (1 - tol) < best_min:
            to_explore.append(i)
    return(to_explore)

def get_NN_one_x(all_X, isolate, max_order = 4, n_epochs_for_orders = 50, n_epochs_for_model = 100, tol_order = 10**(-2), save = True, n_neurons = 256, file = ""):

    """Apply the different routines to get the NN for a isolated X.
        - isolate : index of which x to isolate. int
        - max_order : maximum power to include in the analysis, ie we are going to try to reproduce Y up to
        Y**max_order. int
        - n_epochs_for_orders : numbers of epochs for training when looking what powers should be selected. int
        - n_epochs_for_model : numbers of epochs for training when trying to reproduce Y**orders_to_analyse
        - tol_order : tolerance (in %) in the selection of the different powers to analyse. float
        - save : whether or not to save the data
    """

    X = np.array([all_X[i] for i in range(isolate)]+[all_X[i] for i in range(isolate+1,10)]).T
    Y = all_X[isolate].T
    all_model_this_x = []
    all_losses_this_x = []
    best_losses_this_X = find_order(X, Y, max_order = max_order, n_epochs = n_epochs_for_orders)

    orders_to_analyse = select_order(best_losses_this_X, tol = tol_order)

    print(f"For x{isolate}, the orders to analyse are {np.array(orders_to_analyse)+1}.")
    for this_order in orders_to_analyse:
        print(f"For x{isolate}, analysing order {this_order+1}.")


        model_this_order,loss_this_order = learn_one_x_from_the_others(X, Y, order=this_order+1, n_epochs = n_epochs_for_model, n_neurons = n_neurons)
        all_model_this_x.append([this_order,model_this_order])
        all_losses_this_x.append([this_order,loss_this_order])

        if save == True:
            save_nn(model_this_order,"10d_x"+str(isolate)+"_order"+str(this_order+1), file = file)
    return [all_model_this_x,all_losses_this_x]

def get_all_NN(all_X, max_order = 4, n_epochs_for_orders = 50, n_epochs_for_model = 100, tol_order = 10**(-2), save = True):

    all_models = []
    all_losses = []
    for isolate in range(len(all_X)):
        res = get_NN_one_x(isolate, max_order = max_order, n_epochs_for_orders = n_epochs_for_orders, n_epochs_for_model = n_epochs_for_model, tol_order = tol_order, save = save)
        for i in res:
            all_models.append(i[0])
            all_losses.append(i[1])
    return all_models,all_losses


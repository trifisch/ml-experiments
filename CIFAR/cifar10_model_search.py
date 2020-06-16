import os

# reduce info messages and warnings from TF (may bee needed before tf import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from cifar10_data import CIFAR10
from cifar10_params import nn_params
from cifar10_helper import get_tensorboard_callback, get_model_save_path, save_params, get_model, plot_history, create_params


# reduce info messages and warnings from TF
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# get CIFAR data
# (scale_mode: "standard", "minmax", "none"; augment_size: set to 0 to skip augmentation)
data = CIFAR10(scale_mode="minmax", augment_size=5000)

# auto create parameters
# => comment out if you want to use the parameters from the parameter file!
nn_params = create_params(10)

# Iterate through parameter settings
best_val_accuracy=0.0
count = 0
max_count = len(nn_params)

for params in nn_params:
    count += 1
    print(f"------ Parameter set {count} of {max_count} ------")
    print(str(params) + "\n")

    # Build the model
    model = get_model(data, params["model"])

    # print model info
    #model.summary()

    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x=data.x_train, 
        y=data.y_train, 
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_data=(data.x_val, data.y_val),
        callbacks=[get_tensorboard_callback(), reduce_lr_callback, early_stopping_callback],
        use_multiprocessing=False)

    val_accuracy = history.history["val_accuracy"][-1]
    print(f"*** Model (params[{count-1}] >{params['name']}<): accuracy={history.history['accuracy'][-1]:.4f}, val_accuracy={val_accuracy:.4f}, ")

    if val_accuracy > best_val_accuracy:
        # new best model found => print message, save model and its paramaters
        print(f"****** New best model! ******")
        path = get_model_save_path()
        model.save(path)
        header = f"*** Model (params[{count-1}] >{params['name']}<): accuracy={history.history['accuracy'][-1]:.4f}, val_accuracy={val_accuracy:.4f}, "
        save_params(path, params, header)

        best_val_accuracy = val_accuracy
        # visualize learning progress in (so far) best model
        #plot_history(history)
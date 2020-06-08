import os
import tensorflow as tf

from cifar10_data import CIFAR10
from cifar10_params import nn_params
from cifar10_helper import get_tensorboard_callback, get_model_save_path, get_model, plot_history


# reduce info messages and warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# get CIFAR data
# (scale_mode: "standard", "minmax", "none"; augment_size: set to 0 to skip augmentation)
data = CIFAR10(scale_mode="minmax", augment_size=0)

# Iterate through parameter settings
best_val_accuracy=1e20
count = 0
max_count = len(nn_params)

for params in nn_params:
    count += 1
    print(f"------ Parameter set {count} of {max_count} ------")

    # Build the model
    model = get_model(data, params["model"])

    # print model info
    #model.summary()

    history = model.fit(
        x=data.x_train, 
        y=data.y_train, 
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_data=(data.x_val, data.y_val),
        callbacks=[get_tensorboard_callback()],
        use_multiprocessing=False)

    if history.history["val_accuracy"][-1] < best_val_accuracy:
        # new best model found => print message and save model
        print(f"****** New best model (params[{count-1}] >{params['name']}<): accuracy={history.history['accuracy'][-1]:.4f}, val_accuracy={history.history['val_accuracy'][-1]:.4f}, ")
        model.save(get_model_save_path())

        # visualize learning progress in (so far) best model
        #plot_history(history)
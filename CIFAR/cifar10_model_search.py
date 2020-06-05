import os
import time
import datetime

import random
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.model_selection import GridSearchCV, ParameterGrid

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from cifar10_data import CIFAR10

# reduce warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# parameters
scale_mode="minmax" # "standard", "minmax", "none"
augment_size=5000   # set to 0 to skip augmentation

do_grid_search = True   # if False, use random search

# *** for grid or random search ***
batch_size = 320 # 256 # 512 # 256 # 128
epochs = 20
learning_rate = 0.0008460466124505797
optimizer = Adam

parameters = {
    "epochs": [20],
    "optimizer": [Adam],    #, RMSprop] -> Adam has a good track record and is recommended in many places, therefore sticking with it for the moment
    "learning_rate": [0.0008460]  # 0.00084 was good / [random.uniform(1e-4, 1e-3) for _ in range(100)]
}

# run 1 (BEST)
# Best: 0.635924 using {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.0008460466124505797}
# Acc: 0.635924 (+/- 0.020522) with: {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.0008460466124505797}

# run 2
# Best: 0.629600 using {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.0008}
# Acc: 0.629600 (+/- 0.016658) with: {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.0008}

# get data
cifar = CIFAR10(scale_mode, augment_size)

x_train, y_train = cifar.x_train, cifar.y_train
x_val, y_val = cifar.x_val, cifar.y_val
x_test, y_test = cifar.x_test, cifar.y_test
num_classes = cifar.num_classes

# *** setup logging ***

# postfix for model file
postfix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
time_str = str(time.time())
# Save Path
model_dir = os.path.abspath("/Users/marco/projects/DL/TFKurs/_data/models/")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = os.path.join(model_dir, "mz_model_" + postfix + ".h5")
# Log path
log_dir = os.path.abspath("/Users/marco/projects/DL/TFKurs/_data/logs/")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_dir = os.path.join(log_dir, "cifar10/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True)


# *** The Model Function ***
def model_fn(optimizer, learning_rate):
    # Define the CNN
    input_img = Input(shape=x_train.shape[1:])

    # Orig: filters = 32, 64, 64 / kernels = 3, 5, 5
    # Run 1: filters = 92, 64, 32 / kernels = 5, 5, 3
    x = Conv2D(filters=64, kernel_size=5, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=164, kernel_size=3, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=164, kernel_size=3, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"])
    return model


# *** The Search Algorithms ***

# *** RANDOM ***
def randomized_search():
    keras_clf = KerasClassifier(
        build_fn=model_fn,
        batch_size=batch_size,
        verbose=1)

    rand_cv = RandomizedSearchCV(
        estimator=keras_clf,
        param_distributions=parameters,
        n_iter=4,
        n_jobs=-1,           # use 1 if parallelism does not work
        verbose=1,
        cv=3)

    rand_result = rand_cv.fit(x_train, y_train)

    # Summary
    print("Best: %f using %s" % (rand_result.best_score_, rand_result.best_params_))

    means = rand_result.cv_results_["mean_test_score"]
    stds = rand_result.cv_results_["std_test_score"]
    params = rand_result.cv_results_["params"]

    for mean, std, param in zip(means, stds, params):
        print("Acc: %f (+/- %f) with: %r" % (mean, std, param))

# *** GRID ***
def grid_search():

    grid = ParameterGrid(parameters)
    print("Combinations:")
    for comb in grid:
        print(comb)
    print("<<<")

    keras_clf = KerasClassifier(
        build_fn=model_fn,
        batch_size=batch_size,
        verbose=1)

    grid_cv = GridSearchCV(
        estimator=keras_clf,
        param_grid=parameters,
        n_jobs=-1,              # use 1 if parallelism does not work
        verbose=1,
        cv=3)

    grid_result = grid_cv.fit(x_train, y_train)

    # Summary
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]

    for mean, std, param in zip(means, stds, params):
        print("Acc: %f (+/- %f) with: %r" % (mean, std, param))

    print("Best estimator:")
    print(grid_result.best_estimator_)

# *** Simple search ***
def simple_search():
    # Build the model
    model = model_fn(optimizer, learning_rate)

    # Compile and train (fit) the model, afterwards evaluate the model
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    
    model.fit(
        x=x_train, 
        y=y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[tb],
        use_multiprocessing=False)

    score = model.evaluate(
        x_test, 
        y_test, 
        verbose=1)
    print("Score: ", score)

    # best model
    model.save_weights(filepath=model_path)

simple_search()

#grid_search()


# loading with:
# model.load_weights(filepath=model_path)



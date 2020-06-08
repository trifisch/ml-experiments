import os
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Flatten
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt


def get_tensorboard_callback():
    log_dir = os.path.abspath("./CIFAR/_logs/")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True)

    return tb


def get_model_save_path():
    model_dir = os.path.abspath("./CIFAR/_models/")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(
        model_dir, "model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return model_path


# *** The Model Function ***
def get_model(data, params):
    input = Input(shape=data.x_train.shape[1:])

    x = input
    for element in params["network"]:
        if element[0] == "C2D":
            x = Conv2D(filters=element[1], kernel_size=element[2], padding='same')(x)
            if element[3]:
                x = BatchNormalization()(x)
        elif element[0] == "Dense":
            x = Dense(units=element[1])(x)
        elif element[0] == "A":
            x = Activation(element[1])(x)
        elif element[0] == "MaxPool2D":
            x = MaxPool2D()(x)
        elif element[0] == "Flatten":
            x = Flatten()(x)
        else:
            print("Invalid element: " + element[0])

    # There has to be a Dense layer at the end
    x = Dense(units=data.num_classes)(x)
  
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(inputs=[input], outputs=[y_pred])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=params["optimizer"](learning_rate=params["learning_rate"]),
        metrics=["accuracy"])

    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2)
    plt.tight_layout()

    # summarize history for accuracy
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].title('model accuracy')
    axes[0].ylabel('accuracy')
    axes[0].xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].title('model loss')
    axes[1].ylabel('loss')
    axes[1].xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper left')
    
    plt.show()

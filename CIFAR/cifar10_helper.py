import os
import datetime
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Flatten
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import *
from matplotlib import pyplot as plt

PARAM_FILE = 'params.txt'

# Also log the learning rate in TensorBoard
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)


def get_tensorboard_callback():
    log_dir = os.path.abspath("./CIFAR/_logs/")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tb = LRTensorBoard(
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

# save given parameters to file
def save_params(path, params, header):
    out_str = header + "\n"
    out_str += '"name": "' + params["name"] + "\n"
    out_str += '"epochs": ' + str(params["epochs"]) + '"\n'
    out_str += '"batch_size": "' + str(params["batch_size"]) + '"\n'
    out_str += '"model":\n'

    model_params = params["model"]
    out_str += '"optimizer": "' + str(model_params["optimizer"]) + '"\n'
    out_str += '"learning_rate": ' + str(model_params["learning_rate"]) + '\n'
    
    out_str += '"network":\n'
    for element in model_params["network"]:
        if element[0] == "C2D":
            out_str += str(element[0]) + ", " + str(element[1]) + ", " + str(element[2]) + ", " + str(element[3]) + "\n"
        elif element[0] == "Dense":
            out_str += str(element[0]) + ", " + str(element[1]) + "\n"
        elif element[0] == "A":
            out_str += str(element[0]) + ", " + str(element[1]) + "\n"
        elif element[0] == "MaxPool2D":
            out_str += str(element[0]) + "\n"
        elif element[0] == "Flatten":
            out_str += str(element[0]) + "\n"
        else:
            out_str += "Invalid element\n"

    if not os.path.exists(path):
        os.mkdir(path)
    output_file = os.path.join(path, PARAM_FILE)

    with open(output_file, 'w') as output:
        output.write(out_str)  # save as text
        output.close()


# create random parameters
def create_params(number):
    params = []

    for i in range(number):
        param = {}
        param["name"] = str(i+1)
        param["epochs"] = 30
        param["batch_size"] = 128
        
        model = {}
        model["optimizer"] = Adam
        model["learning_rate"] = np.random.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

        # create random network
        network = [] 
        nr_conv_layers = np.random.choice([1, 2, 3, 4, 5])
        double_layers = np.random.choice([True, False])

        for j in range(nr_conv_layers):
            filters = np.random.choice([64, 128, 256, 512, 1024])
            kernel_size = np.random.choice([3, 5, 7, 9])
            batch_norm = np.random.choice([True, False])

            conv_layer = ["C2D", filters, (kernel_size, kernel_size), batch_norm]
            network.append(conv_layer)
            network.append(["A", "relu"])

            if double_layers:
                network.append(conv_layer)
                network.append(["A", "relu"])
            
            network.append(["MaxPool2D"])

        # ["Flatten"], ["Dense", 140], ["A", "relu"]
        network.append(["Flatten"])
        dense_size = np.random.choice([64, 128, 256, 512, 1024])
        network.append(["Dense", dense_size])
        network.append(["A", "relu"])

        model["network"] = network
        param["model"] = model
        
        params.append(param)
    
    return params



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

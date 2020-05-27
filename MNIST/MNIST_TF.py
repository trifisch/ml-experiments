import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from plotting import display_sample_digits
from plotting import display_sample_digits_validation

# avoid CPU warning ("Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAVE_PATH = os.path.expanduser('~/projects/ML-Experiments/MNIST/saved_models/')
W1_FILE = SAVE_PATH + 'NN_W1.txt'
W2_FILE = SAVE_PATH + 'NN_W2.txt'
B1_FILE = SAVE_PATH + 'NN_b1.txt'
B2_FILE = SAVE_PATH + 'NN_b2.txt'

#**************************************************************************************
#* Model variables
NUM_CLASSES = 10    # we have the digits 0-9 => 10 classes
LOAD_MODEL = True # if True, then we load weights and biases from file

epochs = 0 if LOAD_MODEL else 1000

learning_rate = 0.01
hidden_layer_size = 80
num_features = 28 * 28
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
metric_fn = tf.metrics.CategoricalAccuracy()
loss_fn = tf.losses.CategoricalCrossentropy()
W1, W2, b1, b2 = None, None, None, None

#**************************************************************************************
#* Support functions

def initialize_weights_and_biases():
    global W1, W2, b1, b2

    # Initialize weights for first layer and output layer with random values (mean=0.0, stddev=1.0)
    W1 = tf.Variable(tf.random.truncated_normal(shape=[num_features, hidden_layer_size], dtype=tf.float32, stddev=0.1), name="W1")
    W2 = tf.Variable(tf.random.truncated_normal(shape=[hidden_layer_size, NUM_CLASSES], dtype=tf.float32, stddev=0.1), name="W2")

    # Biases (Vectors)
    b1 = tf.Variable(tf.constant(0.0, shape=[hidden_layer_size]), name="b1")
    b2 = tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]), name="b2")


def load_weights_and_biases():
    global W1, W2, b1, b2

    W1_values = np.loadtxt(W1_FILE).astype(np.float32)
    W1 = tf.Variable(W1_values, name="W1")
    W2_values = np.loadtxt(W2_FILE).astype(np.float32)
    W2 = tf.Variable(W2_values, name="W2")
    b1_values = np.loadtxt(B1_FILE).astype(np.float32)
    b1 = tf.Variable(b1_values, name="b1")
    b2_values = np.loadtxt(B2_FILE).astype(np.float32)
    b2 = tf.Variable(b2_values, name="b2")



def save_weights_and_biases():
    global W1, W2, b1, b2

    np.savetxt(W1_FILE, W1.numpy())
    np.savetxt(W2_FILE, W2.numpy())
    np.savetxt(B1_FILE, b1.numpy())
    np.savetxt(B2_FILE, b2.numpy())


def predict(x):
    """ Predict y (one hot) for given x. y is a matrix with one hot vectors """
    input_layer = x
    
    # calculate hiddenlayer (based on input layer)
    hidden_layer = tf.math.add(tf.linalg.matmul(input_layer, W1), b1)
    hidden_layer_act = tf.nn.sigmoid(hidden_layer)  # orig: relu(hidden_layer)

    # calculate output layer (based on hidden layer)
    output_layer = tf.math.add(tf.linalg.matmul(hidden_layer_act, W2), b2)
    output_layer_act = tf.nn.softmax(output_layer)

    return output_layer_act


def compute_metrics(x,y):
    """ Compute metrics for given x, y """
    y_pred = predict(x)
    metric_fn.update_state(y, y_pred)
    metric_val = metric_fn.result()
    metric_fn.reset_states()
    return metric_val

#**************************************************************************************
#* Prepare dataset

# get MNIST training and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# get training and test set dimensions
train_size = x_train.shape[0]
test_size = x_test.shape[0]
picture_dimension = (x_train.shape[1], x_train.shape[2])

# Reshape the input data ("flatten" the pictures into vectors)
x_train = x_train.reshape(train_size, num_features)
x_test = x_test.reshape(test_size, num_features)

# convert data from int to float in interval 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Cast to np.float32, to ensure that TF runs properly
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Convert given result values into categorical classes (because we are doing classification)
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Visualize a subset of the training set for validation purposes
#display_sample_digits(x_train, picture_dimension, 10, 10)

#**************************************************************************************
#* Create the model

if LOAD_MODEL:
    load_weights_and_biases()
else:
    initialize_weights_and_biases()

# List of all variables in our neural network (these are the values which will be optimized)
nn_variables = [W1, W2, b1, b2]

#**************************************************************************************
#* Train the model iteratively

# create empty lists to save the loss and metric values
train_losses, train_metrics = [], []
test_losses, test_metrics = [], []

# iterate over the number of epochs
for epoch in range(epochs):
    # compute the loss and the metric value for the train set
    # -> train_loss = self.update_variables(x_train, y_train).numpy()
    with tf.GradientTape() as g:
        # watch our weights and biases to allow gradient calculations later
        g.watch(nn_variables)
        
        # *** FORWARD PROPAGATION ***
        y_pred = predict(x_train)

        # Computes the loss (error) between the given training labels and predictions
        loss = loss_fn(y_train, y_pred)

    # *** BACKWARD PROPAGATION ***
    gradients = g.gradient(loss, nn_variables)
    optimizer.apply_gradients(zip(gradients, nn_variables)) 

    # *** Compute metrics... ***
    # ... for training set
    train_loss = loss.numpy()
    train_metric = compute_metrics(x_train, y_train).numpy()
    train_losses.append(train_loss)
    train_metrics.append(train_metric)

    # ... for test set
    test_loss = loss_fn(y_test, predict(x_test)).numpy()
    test_metric = compute_metrics(x_test, y_test).numpy()
    test_losses.append(test_loss)
    test_metrics.append(test_metric)
    
    # Print metrics
    if epoch%10 == 0:
        print("Epoch: ", epoch+1, " of ", epochs,
                " - Train Loss: ", round(train_loss, 4),
                " - Train Metric: ", round(train_metric, 4),
                " - Test Loss: ", round(test_loss, 4),
                " - Test Metric: ", round(test_metric, 4))


# TODO: After running through the epochs, we visualize the loss and metric values
#display_convergence_error(train_losses, test_losses)
#display_convergence_acc(train_metrics, test_metrics)

# Save calculated weights and biases
save_weights_and_biases()

#**************************************************************************************
#* Show validation image as result

# Show randomly selected digits with their prediction for checking purposes
# Visualize a subset of the training set for validation purposes

sample_digits_rows = sample_digits_cols = 10
num_samples = sample_digits_rows * sample_digits_cols

# select sample digits from given test set
sample_digits = np.random.randint(x_test.shape[0] , size=num_samples)

x_test_samples = x_test[sample_digits]
y_test_samples = y_test[sample_digits]
y_pred_samples = predict(x_test_samples).numpy()

# convert ys from one hot into digit
y_test_samples = np.argmax(y_test_samples, axis=1)
y_pred_samples = np.argmax(y_pred_samples, axis=1)

display_sample_digits_validation(x_test_samples, y_test_samples, y_pred_samples, picture_dimension, sample_digits_rows, sample_digits_cols)



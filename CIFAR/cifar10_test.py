import os


# reduce info messages and warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from cifar10_data import CIFAR10


model_path = os.path.abspath("CIFAR/_test/model_20200609-135404")

# reduce info messages and warnings from TF
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "n/a"]

def display_sample_pics_validation(x_test_samples, y_test_samples, y_pred_samples, dim, sample_img_rows=1, sample_img_cols=1):

    # number of samples to show
    num_samples = sample_img_rows * sample_img_cols

    fig=plt.figure(figsize=(8, 8))

    # paint all images
    for i in range(0, num_samples):
        # get the image and re-shape it to the given dimensions
        img = x_test_samples[i].reshape(dim)
        axes = fig.add_subplot(sample_img_rows, sample_img_cols, i+1)
        # axes.set_axis_off()
        axes.tick_params(labelbottom = False, labeltop = False, labelleft = False, labelright = False,
                        bottom = False, top = False, left = False, right = False)
                        
        if y_test_samples[i] != y_pred_samples[i]:
            color = "red"
        else:
            color = "black"
        axes.set_title(cifar10_classes[y_test_samples[i]] + " -> " + cifar10_classes[y_pred_samples[i]], fontsize=8, y=1.0, color=color)
            
        plt.imshow(img) #, cmap=plt.get_cmap('gray_r'))

    # leave more space between the digit images
    plt.tight_layout()
    plt.subplots_adjust(hspace = .54)   # correct spacing found out with plt.subplot_tool()

    plt.show()


# Load CIFAR data (only test set is used here)
data = CIFAR10(scale_mode="minmax")

# Load model
model = load_model(filepath=model_path)

# Evaluate model
score = model.evaluate(
    data.x_test,
    data.y_test,
    verbose=1)

print("Score: ", score)

#**************************************************************************************
# Show randomly selected test images with their prediction for checking purposes

sample_image_rows = sample_image_cols = 7
num_samples = sample_image_rows * sample_image_cols

# select sample images from given test set
samples = np.random.randint(data.x_test.shape[0] , size=num_samples)

x_test_samples = data.x_test[samples]
y_test_samples = data.y_test[samples]
y_pred_samples = model.predict(x_test_samples)

# convert ys from one hot into digit
y_test_samples = np.argmax(y_test_samples, axis=1)
y_pred_samples = np.argmax(y_pred_samples, axis=1)

picture_dimension = (data.x_test.shape[1], data.x_test.shape[2], data.x_test.shape[3])
display_sample_pics_validation(x_test_samples, y_test_samples, y_pred_samples, picture_dimension, sample_image_rows, sample_image_cols)


import os
import tensorflow as tf

from tensorflow.keras.models import load_model
from cifar10_data import CIFAR10


model_path = os.path.abspath("/Users/marco/projects/ML-Experiments/CIFAR/_models/model_20200608-182053")

# reduce info messages and warnings from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

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

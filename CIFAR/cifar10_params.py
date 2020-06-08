from tensorflow.keras.optimizers import *

# Parameter remarks
# Batch size: range may be [32, 512]? 320 had good results, but article suggests that lower batch sizes may be better
# Optimizer: Original choice was between Adam and RMSprop.
#            -> Adam has a good track record and is recommended in many places, therefore sticking with it for the moment
# Network: Consists of blocks describing the neural network elements
#            Conv2D (C2D): filters, kernel_size, useBatchNormalization
#            Activation (A): activation function
nn_params = [
    {
        "name": "1",
        "epochs": 2,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,  
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
#                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
#                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },
    {
        "name": "2",
        "epochs": 2,
        "batch_size": 256,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
#                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    }
]
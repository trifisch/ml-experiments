from tensorflow.keras.optimizers import *

# Parameter remarks
# Batch size: range may be [32, 512]? 320 had good results, but article suggests that lower batch sizes may be better
# Optimizer: Original choice was between Adam and RMSprop.
#            -> Adam has a good track record and is recommended in many places, therefore sticking with it for the moment
# Network: Consists of blocks describing the neural network elements
#            Conv2D (C2D): filters, kernel_size, useBatchNormalization
#            Activation (A): activation function
nn_params = [
    # ------ Based on parameter set 2 of v1 ------
    # with epochs = 20, augment_size=0, + reduce_lr_callback, early_stopping_callback
    # *** Model (params[0] >Best v1 - orig<): accuracy=1.0000, val_accuracy=0.7788,
    # with epochs = 40, augment_size=5000, + reduce_lr_callback, early_stopping_callback
    #*** Model (params[0] >Best v1 - orig<): accuracy=1.0000, val_accuracy=0.7663,
    {
        "name": "Best v1 - orig",
        "epochs": 40,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    }
]

nn_params_v2 = [
    # ------ Based on parameter set 2 of v1 ------
    # ****** New best model (params[0] >Best v1 - orig<): accuracy=0.9700, val_accuracy=0.7428,
    {
        "name": "Best v1 - orig",
        "epochs": 20,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },
    #****** New best model (params[1] >Best v1 - incr kernel size<): accuracy=0.9606, val_accuracy=0.6800,
    {
        "name": "Best v1 - incr kernel size",
        "epochs": 20,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 7, False],  ["A", "relu"], ["C2D", 64, 7, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 7, False],  ["A", "relu"], ["C2D", 128, 7, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 5, False],  ["A", "relu"], ["C2D", 164, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },
    # ****** New best model (params[2] >Best v1 - incr kernel + larger layer<): accuracy=0.9643, val_accuracy=0.6653,
    {
        "name": "Best v1 - incr kernel + larger layer",
        "epochs": 20,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 128, 7, False],  ["A", "relu"], ["C2D", 128, 7, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 7, False],  ["A", "relu"], ["C2D", 164, 7, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 212, 5, False],  ["A", "relu"], ["C2D", 212, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },
    #****** New best model (params[3] >Best v1 - bacth normalized<): accuracy=0.9791, val_accuracy=0.7395,
    {
        "name": "Best v1 - bacth normalized",
        "epochs": 20,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, True],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, True],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, True],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },
]

nn_params_v1 = [
    # ------ Parameter set 1 of 5 ------
    # ****** New best model (params[0] >1<): accuracy=0.9663, val_accuracy=0.7149,
    {
        "name": "1",
        "epochs": 20,
        "batch_size": 64,

        "model": {
            "optimizer": Adam,  
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },

    # ------ Parameter set 2 of 5 ------
    # ****** New best model (params[1] >2<): accuracy=0.9735, val_accuracy=0.7534,
    {
        "name": "2",
        "epochs": 20,
        "batch_size": 128,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },

    # ------ Parameter set 3 of 5 ------
    # ****** New best model (params[2] >3<): accuracy=0.9746, val_accuracy=0.7443,
    {
        "name": "3",
        "epochs": 20,
        "batch_size": 256,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 64, 5, False],  ["A", "relu"], ["C2D", 64, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 164, 3, False],  ["A", "relu"], ["C2D", 164, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },

    # ------ Parameter set 4 of 5 ------
    # ****** New best model (params[3] >4<): accuracy=0.9646, val_accuracy=0.7179,
    {
        "name": "4",
        "epochs": 20,
        "batch_size": 64,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 194, 5, False],  ["A", "relu"], ["C2D", 194, 5, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 256, 3, False],  ["A", "relu"], ["C2D", 256, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    },

    # ------ Parameter set 5 of 5 ------
    # ****** New best model (params[4] >5<): accuracy=0.9689, val_accuracy=0.6920,
    {
        "name": "5",
        "epochs": 20,
        "batch_size": 64,

        "model": {
            "optimizer": Adam,
            "learning_rate": 0.0008460,

            "network": [
                ["C2D", 128, 5, False],  ["A", "relu"], ["C2D", 128, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 194, 5, False],  ["A", "relu"], ["C2D", 194, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["C2D", 256, 3, False],  ["A", "relu"], ["C2D", 256, 3, False],  ["A", "relu"], ["MaxPool2D"],
                ["Flatten"], ["Dense", 128], ["A", "relu"]
            ]
        }
    }
]
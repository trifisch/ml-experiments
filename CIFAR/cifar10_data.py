import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# configurable parameters
val_ratio = 0.3 # percentage of validation set values
random_state = 0 # for reproducible splits
rotation_range=10
zoom_range=0.05
width_shift_range=0.05
height_shift_range=0.05
fill_mode='constant'
cval=0.0
horizontal_flip=False
vertical_flip=False


class CIFAR10:
    def __init__(self, scale_mode="none", augment_size=0):
        # Load the data set
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

        # split validation set from training set
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=val_ratio, random_state=random_state)

        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.val_size = self.x_val.shape[0]
        self.test_size = self.x_test.shape[0]

        # image dimensions
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_features = self.width * self.height * self.depth

        self.num_classes = 10 # Constant for the data set
        
        # Reshape the y data to one hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_val = to_categorical(self.y_val, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)

        # augment train data
        self.augment_data(augment_size=augment_size)

        # scale train, val and test data
        self.scale_data(scale_mode=scale_mode)


    def augment_data(self, augment_size=0):
        if augment_size==0:
            return

        # Create an instance of the image data generator class
        image_generator = ImageDataGenerator(
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            fill_mode=fill_mode,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            cval=cval)

        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)

        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(x_augmented, batch_size=augment_size, shuffle=False).next()#[0]

        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

    def scale_data(self, scale_mode="none", preprocess_params=None):
        # Preprocess the data
        if scale_mode == "standard":
            if preprocess_params:
                self.scaler = StandardScaler(**preprocess_params)
            else:
                self.scaler = StandardScaler()
        elif scale_mode == "minmax":
            if preprocess_params:
                self.scaler = MinMaxScaler(**preprocess_params)
            else:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            return

        # Temporary flatteining of the x data
        self.x_train = self.x_train.reshape(self.train_size, self.num_features)
        self.x_val = self.x_val.reshape(self.val_size, self.num_features)
        self.x_test = self.x_test.reshape(self.test_size, self.num_features)
        
        # Fitting and transforming
        self.scaler.fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        # Reshaping the xdata back to the input shape
        self.x_train = self.x_train.reshape(
            (self.train_size, self.width, self.height, self.depth))
        self.x_val = self.x_val.reshape(
            (self.val_size, self.width, self.height, self.depth))
        self.x_test = self.x_test.reshape(
            (self.test_size, self.width, self.height, self.depth))

# just for general check: show random pics from train, val, and test set
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cifar = CIFAR10(scale_mode="minmax")
    
    train_img = cifar.x_train[np.random.randint(0, cifar.x_train.shape[0])]
    val_img = cifar.x_train[np.random.randint(0, cifar.x_val.shape[0])]
    test_img = cifar.x_train[np.random.randint(0, cifar.x_test.shape[0])]

    fig, axes = plt.subplots(1, 3)
    plt.tight_layout()
    scale = 1.0 # use 255.0 if not already scaled upfront, e.g. with minmax scaler
    axes[0].imshow(train_img/scale)   
    axes[0].set_title("Train")
    axes[1].imshow(val_img/scale)
    axes[1].set_title("Val")
    axes[2].imshow(test_img/scale)
    axes[2].set_title("Test")
    plt.show()

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Display the Digit from the image
# If the Label and PredLabel is given display it too
def display_sample_digits(digits, dim, sample_digits_rows = 1, sample_digits_cols = 1):

    # number of samples to show
    num_samples = sample_digits_rows * sample_digits_cols

    # select samples from given digits
    sample_digits = np.random.randint(digits.shape[0] , size=num_samples)

    # create sample_digits_rows x sample_digits_cols subplots to show all digits in a nice grid
    fig=plt.figure(figsize=(8, 8))

    # paint all images
    for i in range(0, num_samples):
        # get the randomly selected image and re-shape it to the given dimensions
        img = digits[sample_digits[i]].reshape(dim)
        axes = fig.add_subplot(sample_digits_rows, sample_digits_cols, i+1)
        # axes.set_axis_off()
        axes.tick_params(labelbottom = False, labeltop = False, labelleft = False, labelright = False,
                        bottom = False, top = False, left = False, right = False)
        plt.imshow(img, cmap=plt.get_cmap('gray_r'))

    plt.show()


def display_sample_digits_validation(x_test_samples, y_test_samples, y_pred_samples, dim, sample_digits_rows=1, sample_digits_cols=1):

    # number of samples to show
    num_samples = sample_digits_rows * sample_digits_cols

    # create sample_digits_rows x sample_digits_cols subplots to show all digits in a nice grid
    fig=plt.figure(figsize=(8, 8))

    # paint all images
    for i in range(0, num_samples):
        # get the image and re-shape it to the given dimensions
        img = x_test_samples[i].reshape(dim)
        axes = fig.add_subplot(sample_digits_rows, sample_digits_cols, i+1)
        # axes.set_axis_off()
        axes.tick_params(labelbottom = False, labeltop = False, labelleft = False, labelright = False,
                        bottom = False, top = False, left = False, right = False)
                        
        if y_test_samples[i] != y_pred_samples[i]:
            axes.set_title(str(y_test_samples[i]) + " -> " + str(y_pred_samples[i]), fontsize=8, y=0.87, color="red")
        else:
            axes.set_title(str(y_test_samples[i]) + " -> " + str(y_pred_samples[i]), fontsize=8, y=0.87)
            
        plt.imshow(img, cmap=plt.get_cmap('gray_r'))

    # leave ore space between the digit images
    plt.tight_layout()

    plt.show()

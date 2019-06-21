import keras
import numpy as np

from keras.datasets import cifar10
from keras.models import load_model
from keras.preprocessing.image import save_img
from scipy.ndimage.filters import gaussian_filter

import pandas as pd
import matplotlib.pyplot as plt

# Data information
total_images = 2000 # NOT USED
num_classes = 10
model = load_model('saved_models/keras_cifar4_four_model.h5')

# Gaussian blur settings
std_dev = 0
std_dev_delta = 0.1
std_dev_max = 3

# Plot data arrays
std_dev_arr = []
accuracy_arr = []

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Filters unwanted unwanted images
xl_test = x_test.tolist()
yl_test = y_test.tolist()

cnt = 0
for idx, i in enumerate(list(yl_test)):
    if (i[0] != 2 and i[0] != 6 and i[0] != 3 and i[0] != 5): # Set for 4
        yl_test.remove(i)
        xl_test.pop(cnt)
    else:
        cnt += 1

x_test = np.asarray(xl_test)
y_test = np.asarray(yl_test)

# Appropriately transforms data for the model
y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255

# Applies gaussian_filter on training images and tests model
for x in np.arange(std_dev, std_dev_max+std_dev_delta, std_dev_delta):
    print('Gaussian test ' + str(x) + '/' + str(std_dev_max))
    std_dev_arr.append(x)
    gauss = gaussian_filter(x_test, x)

    # Score trained model on blurred images
    scores = model.evaluate(gauss, y_test, verbose=1)
    accuracy_arr.append(scores[1]) # Adds accuracy

df = pd.DataFrame({'x': std_dev_arr, 'y': accuracy_arr})
plt.plot('x', 'y', data=df, linestyle='', marker='o')
plt.xlabel('Gaussian Blur Standard Deviation')
plt.ylabel('Accuracy Percentage')
plt.savefig('images/gauss-four.jpg')

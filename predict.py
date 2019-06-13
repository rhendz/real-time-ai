from keras.preprocessing import image
from keras.models import load_model
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import time

total_images = 22166

std_dev = 0
std_dev_delta = 0.1
std_dev_max = 3

# Plot data
std_dev_arr = []
accuracy_arr = []

classifier = load_model('classifier.h5')

file_path = 'dataset/validation/test'

for x in np.arange(std_dev, std_dev_max+std_dev_delta, std_dev_delta):
    std_dev_arr.append(x)

    counter = 1

    # load all images into a list
    images = []
    actual = [] # Keeps track of actual classification

    print('Testing std_dev', x)
    sys.stdout.flush()
    for img in os.listdir(file_path):
        if img.find('cat') >= 0:
            actual.append(0)
        else:
            actual.append(1)
        img = image.load_img(os.path.join(file_path, img), target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = gaussian_filter(img, x)
        images.append(img)

        # print('Testing', counter, '/', total_images)
        counter+=1

    # stack up images list to pass for prediction
    images = np.vstack(images)

    classes = classifier.predict_classes(images, batch_size=10)
    actual = np.vstack(actual)

    accuracy = np.sum(np.equal(actual, classes)) / total_images
    accuracy_arr.append(accuracy)

df = pd.DataFrame({'x': std_dev_arr, 'y': accuracy_arr})
plt.plot('x', 'y', data=df, linestyle='', marker='o')
plt.xlabel('Gaussian Blur Standard Deviation')
plt.ylabel('Accuracy Percentage')
plt.savefig('gauss2.jpg')

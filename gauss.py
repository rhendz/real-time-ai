from keras.preprocessing import image
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import os

file_path = 'dataset/validation/test'

img = image.load_img(os.path.join(file_path, 'cat.2.jpg'), target_size=(64, 64))
img = image.img_to_array(img)

blurred = gaussian_filter(img, 3)

# Convert blurred image back to image and show
img = image.array_to_img(blurred)
img.show()

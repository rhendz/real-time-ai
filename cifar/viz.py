from keras.utils import plot_model
from keras.models import load_model

model = load_model('saved_models/keras_cifar4_four_model.h5')

plot_model(model, to_file='images/model.jpg', show_shapes=True)

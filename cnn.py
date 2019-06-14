# Convolutional neural network

# Keras libraries and packages
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model

# Image processing
from keras.preprocessing.image import ImageDataGenerator

# Display
from IPython.display import display
from PIL import Image

# Initialize the CNN
classifier = Sequential()

# Applies convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Applies pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Applies flattening
classifier.add(Flatten())

# Connects the network
classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 1))

# Compiles the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Model loading if stopped midway
# classifier.load_weights('tmp/weights-03.h5')

# Fits the CNN to the images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
        )

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
        )

checkpointer=ModelCheckpoint('tmp/weights-{epoch:02d}.h5', verbose=1, save_best_only=True)

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_set,
        validation_steps=800,
        use_multiprocessing=True,
        workers=24,
        callbacks=[checkpointer]
        )

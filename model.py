import os
import math
import numpy as np
from skimage.io import imread
from tensorflow.python.estimator import keras
from keras import layers
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from add_func import masks_as_image


from constants import TRAIN_FOLDER,TEST_FOLDER,CSV_PATH,MODEL_PATH, IMG_SIZE,NUM_CLASSES,BATCH_SIZE,MAX_TRAIN_STEPS,VALIDATION_SPLIT,NB_EPOCHS,RANDOM_STATE,VAL_IMAGES

class DataGenerator(Sequence):
    def __init__(self, images_set, masks_set, batch_size):
        self.images, self.masks = images_set, masks_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        return (
                np.array([
                    imread(os.path.join(TRAIN_FOLDER, img_name))
                    for img_name in batch_images]),
                np.array([
                    masks_as_image(self.masks[self.masks['ImageId'] == img_name]['EncodedPixels'])
                    for img_name in batch_images]))


def unet(input_shape):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path (left side of the U-Net)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom of the U-Net
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Expanding path (right side of the U-Net)
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv7)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    return model
model = unet(IMG_SIZE)



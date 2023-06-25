from model import unet,DataGenerator
from dataloader import train_ships, valid_ships,df
import numpy as np
from tensorflow.python.estimator import keras
from keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from model import unet, model
from constants import BATCH_SIZE, NB_EPOCHS,  MODEL_PATH


train_gen = DataGenerator(np.array(train_ships['ImageId']), df, BATCH_SIZE)
valid_gen = DataGenerator(np.array(valid_ships['ImageId']), df, BATCH_SIZE)


def dice_coef(y_true, y_pred, smooth=1):
    # Reshape the true masks
    y_true = K.cast(y_true, 'float32')
    # Calculate the intersection between predicted and true masks
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    # Calculate the union of predicted and true masks
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    # Calculate the Dice coefficient
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    # Combine binary cross-entropy and negative Dice coefficient
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    # Calculate the true positive rate
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)

def precision(y_true, y_pred):
    # Calculate the precision rate (the proportion of true positive predictions
    # out of all positive predictions)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    # Calculate the recall rate (the proportion of true positive predictions
    # out of all positive samples)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (total_positives + K.epsilon())

def specificity(y_true, y_pred):
    # Calculate the specificity rate (the proportion of true negative
    # predictions out of all negative samples)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    total_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (total_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    # Calculate the F1 score (harmonic mean of precision and recall)
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


model.compile(optimizer="adam",
              loss=dice_p_bce,
              metrics=[dice_coef,
                       'binary_accuracy',
                       true_positive_rate,
                       precision,
                       recall,
                       specificity,
                       f1_score])

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Path to save intermediate and model weights
weight_path="{}_weights.best.hdf5".format('seg_model')

# Save the model after each epoch if the validation loss improved
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

# Reduce the learning rate when the metric has stopped improving
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)

# Stop training when the validation loss has stopped improving
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15)

# Keep track of training history by creating a callbacks list
callbacks_list = [checkpoint, early, reduceLROnPlat]

history = model.fit(train_gen,
                    validation_data=valid_gen,
                    epochs=NB_EPOCHS,
                    callbacks=callbacks_list)
print("Training the model...")

model.load_weights(weight_path)
model.save(MODEL_PATH)
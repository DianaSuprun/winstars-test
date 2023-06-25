import os
from tensorflow.python.estimator import keras
from model import unet
from constants import TEST_FOLDER,MODEL_PATH
import numpy as np
import pandas as pd
from keras.losses import binary_crossentropy
from skimage.io import imread
from skimage.morphology import label
from add_func import multi_rle_encode
from keras import models
from keras.models import Model


model = models.load_model(MODEL_PATH, compile=False)
test_paths = os.listdir(TEST_FOLDER)
out_pred_rows = []

for img_id in test_paths:
    img = imread(os.path.join(TEST_FOLDER, img_id))
    img = np.expand_dims(img, 0)/255.0
    prediction = model.predict(img)[0]
    encodings = multi_rle_encode(prediction)
    # Add an entry with None if there is no ship detected and
    out_pred_rows.append([{'ImageId': img_id, 'EncodedPixels': encoding}
                      if encodings
                      else {'ImageId': img_id, 'EncodedPixels': None}
                      for encoding in encodings])


result_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
result_df.to_csv('result.csv', index=False)


print('user2')
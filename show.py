import cv2 as cv
import tqdm
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import re
model = tf.keras.models.load_model('model.h5', custom_objects={'softmax_v2': tf.nn.softmax}) # モデルの読み込み

while True:
    input()
    img = cv.imread("./test.png")[:,:,0]
    img = np.invert(img)
    img = np.array([img])
    predict=model.predict(img)
    print(f"predicted as {np.argmax(predict[0])} in {str(max(predict[0])*100)}%")

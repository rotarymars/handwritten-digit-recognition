import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('model.h5', custom_objects={'softmax_v2': tf.nn.softmax}) # モデルの読み込み

test_x = []
test_y = []
picturelist = []
for i in os.listdir("./test/"):
    img = cv.imread("./test/" + i)[:, :, 0]
    img = np.invert(img)
    imga = np.array([img])
    test_x.append(img)
    test_y.append(int(i[0]))
    if np.argmax(model.predict(imga)) != int(i[0]):
        picturelist.append(i)
test_x = np.array(test_x)
test_y = np.array(test_y)
loss, accuracy = model.evaluate(test_x, test_y)
print("loss: ", loss)
print("accuracy: ", accuracy)
print(picturelist)

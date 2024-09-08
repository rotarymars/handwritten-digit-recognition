import cv2 as cv
import tqdm
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import re
def comp(a)->bool:
    return int(re.match(r"(.*).png",a).group(1))
    return int(re.match(r"(.*).png",a).group(1))<int(re.match(r"(.*).png",b).group(1))


model = tf.keras.models.load_model('model.h5', custom_objects={'softmax_v2': tf.nn.softmax}) # モデルの読み込み

test_x = []
test_y = []
picturelist = []
for i in tqdm.tqdm(os.listdir("./test/")):
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

for i in picturelist:
    listedfile=os.listdir(f'./data/{i[0]}')
    listedfile=sorted(listedfile, key=comp)
    print(i)
    os.system(f'cp ./test/{i} ./data/{i[0]}/{re.match(r"(.*).png",listedfile[-1]).group(0)}')

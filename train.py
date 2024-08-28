import cv2 as cv
import random
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=200, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
files = []
for i in range(0, 10):
    path = f"./data/{str(i)}/"
    tmpfiles = os.listdir(path)
    tmpfiles = [path + j for j in tmpfiles]
    files = files + tmpfiles
random.shuffle(files)
MAX_PICTURE = 100000 # 何枚の画像を同時にメモリに配置するか
EPOCH = 100 # 何回画像を学習させるか
nowindex = 0 # どこのインデックスまで行ったか
for i in tqdm.tqdm(range(EPOCH)):
    nowindex = 0
    train_x = []
    train_y = []
    while True:
        filepath = files[nowindex]
        img = cv.imread(filepath)[:, :, 0]
        img = np.invert(img)
        train_x.append(img) # 画像を配列に追加
        train_y.append(int(filepath[7])) # なんの数字か配列に追加
        nowindex += 1
        if len(train_x) >= MAX_PICTURE or nowindex >= len(files):
            train_x = np.array(train_x) # numpyの配列に変換
            train_y = np.array(train_y) # 同上
            model.train_on_batch(train_x, train_y) # 学習させる✏
            train_x = []
            train_y = []
        if nowindex >= len(files):
            break
model.save("model.h5") # 作成したモデルを保存

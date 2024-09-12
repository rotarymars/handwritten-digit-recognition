import cv2 as cv
import time
import random
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plotbool = False
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
maxunit = 780
descending = 70
while True:
    if maxunit - descending >= 10:
        print(maxunit-descending)
        model.add(tf.keras.layers.Dense(units=maxunit-descending, activation=tf.nn.relu))
        maxunit-=descending
    else:
        break
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
files = []
for i in range(0, 10):
    path = f"./data/{str(i)}/"
    tmpfiles = os.listdir(path)
    tmpfiles = [path + j for j in tmpfiles]
    files = files + tmpfiles
random.shuffle(files)
MAX_PICTURE = 10000 # 何枚の画像を同時にメモリに配置するか
EPOCH = 100 # 何回画像を学習させるか
nowindex = 0 # どこのインデックスまで行ったか
if plotbool:
    test_x = []
    test_y = []
    for i in os.listdir("./test/"):
        img = cv.imread("./test/" + i)[:, :, 0]
        img = np.invert(img)
        test_x.append(img)
        test_y.append(int(i[0]))
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    plotaccuracy = []
    plotloss = []
    plottime = []
for i in tqdm.tqdm(range(EPOCH)):
    nowindex = 0
    train_x = []
    train_y = []
    if plotbool:
        beforetime = time.time()
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
    if plotbool:
        loss, accuracy = model.evaluate(test_x, test_y)
        plotaccuracy.append(accuracy)
        plotloss.append(loss)
        plottime.append(time.time()-beforetime)
if os.path.isfile("model.h5"):
    os.remove("model.h5")
model.save("model.h5") # 作成したモデルを保存
if plotbool:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    t = range(EPOCH)
    y1 = plotaccuracy
    ln1=ax1.plot(t, y1,'C0',label=r'accuracy')

    ax2 = ax1.twinx()
    y2 = plotloss
    ln2=ax2.plot(t,y2,'C1',label=r'loss')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower right')

    ax1.set_xlabel('t')
    ax1.set_ylabel(r'accuracy')
    ax1.grid(True)
    ax2.set_ylabel(r'loss')
    ax1.set_ylim((max(plotaccuracy)+min(plotaccuracy))/2, 1)
    ax2.set_ylim(min(plotloss),max(plotloss))
    fig.show()
    plt.show()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    t = range(EPOCH)
    y1 = plotaccuracy
    ln1=ax1.plot(t, y1,'C0',label=r'accuracy')

    ax2 = ax1.twinx()
    y2 = plotloss
    ln2=ax2.plot(t,y2,'C1',label=r'time')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower right')

    ax1.set_xlabel('t')
    ax1.set_ylabel(r'accuracy')
    ax1.grid(True)
    ax2.set_ylabel(r'time')
    ax1.set_ylim((max(plotaccuracy)+min(plotaccuracy))/2, 1)
    fig.show()
    plt.show()
exit()

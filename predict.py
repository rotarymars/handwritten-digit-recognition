import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('model.h5', custom_objects={'softmax_v2': tf.nn.softmax})
print(os.getcwd())

firstnum = 0
index = 0
totalcount = 0
accuratecount = 0
x_test = []
y_test = []
for firstnum in range(0, 10):
    index = 0
    while True:
        filepath = './test/' + str(firstnum) + '-' + str(index) + '.png'
        if os.path.isfile(filepath):
            print(filepath + ' found')
            img = cv.imread(filepath)[:,:,0]
            img_train = np.invert(img)
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            x_test.append(img_train)
            y_test.append(firstnum)
            print('predicted ' + str(firstnum) + ' as' + str(np.argmax(prediction)))
            index += 1
        else:
            break


x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = np.array(y_test)
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

'''
for i in range(0, 100):
    filepath = './test/'+str(i)+'.png'
    if os.path.isfile(filepath):
        print(filepath + ' found')
        img = cv.imread(filepath)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(np.argmax(prediction))
        print(prediction)
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    else:
        print(filepath + ' not found')

'''

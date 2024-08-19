import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
def convert_num_to_char(n):
    if n>=0 and n<10:
        return n


x_train = []
y_train = []

nums = "0123456789"
alphas = ""
ALPHAS = ""
nowint = 0
print('\n\n\nfile start scanning')
for i in nums:
    nowint = 0
    path = f'./data/{i}/{str(nowint)}.png'
    if os.path.exists(path):
        while True:
            path = f'./data/{i}/{str(nowint)}.png'
            if os.path.exists(path):
                print(path)
                img = cv.imread(path)[:,:,0]
                img = np.invert(img)
                x_train.append(img)
                y_train.append(int(i))
                print('categorized ' + path + ' as ' + i)
                nowint += 1
            else:
                print('path does not exist')
                break
print('\n\n\nfile scanning end')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
'''
for i in range(100):
    model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
'''

model.add(tf.keras.layers.Dense(units=784, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=700, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=600, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=500, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=400, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=300, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=200, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=len(nums)+len(alphas)+len(ALPHAS), activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # Reshape to include channel dimension
y_train = np.array(y_train)
print(y_train)

model.fit(x_train, y_train, epochs=1000)
model.save('model.h5')
'''
tmp = []

for i in range(11):
    filepath='./data/0/'+str(i)+'.png'
    img = cv.imread(filepath)[:,:,0]
    img = np.invert(np.array([img]))
    print(np.argmax(model.predict(img)))
'''

loss, accuracy = model.evaluate(x_train, y_train)

'''
img = cv.imread('./data/0/0.png')[:,:,0]
img = np.invert(np.array([img]))
prediction = model.predict(img)
print(np.argmax(prediction))
'''
os.system('touch done.file')

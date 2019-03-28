import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras import regularizers

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

tyre_path = 'training_data_prepped/prepped-tyre'
else_path = 'training_data_prepped/prepped-else'

tyre_paths = sorted(glob.glob(os.path.join(tyre_path, '*.jpg')), key=natural_key)
elses_paths = sorted(glob.glob(os.path.join(else_path, '*.jpg')), key=natural_key) 
classnames = ['nontyre','tyre']
width = 15
height = 70
dataset = []
label = []
for path in tyre_paths:
    img = cv2.imread(path)
    img = cv2.resize(img,(width,height))
    img = img_to_array(img)
    dataset.append(img)
    label.append([1,0])

for path in elses_paths:
    img = cv2.imread(path)
    img = cv2.resize(img,(width,height))
    img = img_to_array(img)
    dataset.append(img)
    label.append([0,1])

for path in elses_paths:
    img = cv2.imread(path)
    img = cv2.resize(img,(width,height))
    img = img_to_array(img)
    dataset.append(img)
    label.append([0,1])
dataset = np.array(dataset, dtype="float") / 255.0
label = np.array(label, dtype = 'int')
dataset,label = shuffle(dataset,label)

(trainX, testX, trainY, testY) = train_test_split(dataset,label)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (height,width,3)),
    tf.keras.layers.Dense(2048,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(1024,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(512,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(256,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(128,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(64,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(32,activation = tf.nn.relu,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones()),
    tf.keras.layers.Dense(2,activation = tf.nn.softmax,kernel_initializer = tf.keras.initializers.he_normal(seed=None),bias_initializer = tf.keras.initializers.Ones())
])
model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.03,momentum = 0.1), 
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.binary_accuracy,tf.keras.metrics.mae]
)
history = model.fit(trainX,trainY,epochs=100,batch_size = 40,validation_data=(testX,testY))
test_loss, test_acc, test_mae = model.evaluate(dataset, label)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

fig = plt.figure(figsize=[8,6])

ax1 = fig.add_subplot(121)
ax1.plot(history.history['loss'],'r',linewidth=3.0)
ax1.plot(history.history['val_loss'],'b',linewidth=3.0)
ax1.legend(['Training loss', 'Validation Loss'],loc = 'lower right',fontsize=8)
ax1.set_xlabel('Epochs ',fontsize=16)
ax1.set_ylabel('Loss',fontsize=16)
ax1.set_title('Loss Curves',fontsize=16)

ax2 = fig.add_subplot(122)
ax2.plot(history.history['binary_accuracy'],'r',linewidth=3.0)
ax2.plot(history.history['val_binary_accuracy'],'b',linewidth=3.0)
ax2.legend(['Training Accuracy', 'Validation Accuracy'],loc = 'lower right',fontsize=8)
ax2.set_xlabel('Epochs ',fontsize=16)
ax2.set_ylabel('Accuracy',fontsize=16)
ax2.set_title('Accuracy Curves',fontsize=16)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pylab as plt
import os
import glob
import re
import cv2
from time import time
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

def createModel(inputshape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputshape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
     
    return model











def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

tyre_path = 'training_data_prepped/prepped-tyre'
else_path = 'training_data_prepped/prepped-else'

tyre_paths = sorted(glob.glob(os.path.join(tyre_path, '*.jpg')), key=natural_key)
elses_paths = sorted(glob.glob(os.path.join(else_path, '*.jpg')), key=natural_key) 
classnames = ['nontyre','tyre']
width = 240
height = 560
shape = (height,width,3)
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

model = createModel(shape)
model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.01,momentum = 0.1), 
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.binary_accuracy,tf.keras.metrics.mae]
)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

cb = [tensorboard,tf.keras.callbacks.ModelCheckpoint('./savedmodel/model{epoch:04d}-{val_binary_accuracy:.4f}.h5', monitor='val_binary_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]

history = model.fit(trainX,trainY,epochs = 200,batch_size = 20,validation_data=(testX,testY),callbacks = cb)
test_loss, test_acc, test_mae = model.evaluate(dataset, label)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
model.save('newmodel.h5')

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
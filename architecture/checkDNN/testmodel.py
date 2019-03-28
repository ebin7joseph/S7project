import os
import re
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
import numpy as np





def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

tyre_path = 'training_data_prepped/prepped-tyre'
else_path = 'training_data_prepped/prepped-else'

tyre_paths = sorted(glob.glob(os.path.join(tyre_path, '*.jpg')), key=natural_key)
elses_paths = sorted(glob.glob(os.path.join(else_path, '*.jpg')), key=natural_key) 
classnames = ['nontyre','tyre']
width = 90
height = 210
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

model = tf.keras.models.load_model('savedmodel/high.h5')

print(model.metrics_names)
print(model.evaluate(dataset,label))

print(model.predict(dataset))
print(label)

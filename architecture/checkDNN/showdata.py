import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import cv2
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

 


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

tyre_path = 'training_data_prepped/prepped-tyre'
else_path = 'training_data_prepped/prepped-else'

tyre_paths = sorted(glob.glob(os.path.join(tyre_path, '*.jpg')), key=natural_key)
elses_paths = sorted(glob.glob(os.path.join(else_path, '*.jpg')), key=natural_key) 
classnames = ['nontyre','tyre']
width = 300
height = 700
shape = (height,width,3)
dataset = []
label = []
imag = []
for path in tyre_paths:
    img = cv2.imread(path)
    imag.append(img)
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
dataset = np.array(dataset, dtype="float")
label = np.array(label, dtype = 'int')
dataset,label = shuffle(dataset,label)

plt.figure(figsize=(30,30))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imag[i])
plt.show()
#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install Pillow==2.2.1


# In[16]:


pip install --upgrade tensorflow==1.5.0


# In[10]:


pip install keras==2.3.1


# In[18]:


pip install opencv-python


# In[78]:


import  keras
from sklearn.model_selection import  train_test_split

TEST_DIR='C:/Users/hp/Downloads/Signature_Recognition_DeepLearning-master/Signature_Recognition_DeepLearning-master/data/test/'

SIGNATURE_CLASSES = ['A', 'B', 'C','D','E','F','G','H','Avinash','Shreya','Aniket Shah','Siddharth']
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
#import seaborn as sns
#%matplotlib inline 

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras import backend as K 
import cv2


# In[79]:


ROWS = 190  
COLS = 160 
CHANNELS = 3
TRAIN_DIR='C:/Users/hp/Downloads/Signature_Recognition_DeepLearning-master/Signature_Recognition_DeepLearning-master/data/train/'


# In[80]:


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# In[81]:


def read_image(src):
     """Read and resize individual images"""
     im = cv2.imread(src, cv2.IMREAD_COLOR)
     im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
     return im


# In[82]:


def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images


# In[35]:


files = []
y_all = []

for fish in SIGNATURE_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)
print(len(files))
print(len(y_all))


# In[83]:


X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)
# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)


# In[84]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=12, random_state=23, 
                                                    stratify=y_all)



optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'


# In[85]:


from keras import backend as K
def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)
print('1')


# In[86]:


model = Sequential()

model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 2, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(len(SIGNATURE_CLASSES)))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error)


early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
       
model.fit(X_train, y_train, batch_size=64, nb_epoch=3,
              validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])
preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))


# In[87]:


test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files): 
    test[i] = read_image(TEST_DIR+im)
    
test_preds = model.predict(test, verbose=1)
submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()

submission.to_csv('C:/Users/hp/Downloads/Signature_Recognition_DeepLearning-master/Signature_Recognition_DeepLearning-master/signatureResults.csv',index=False)


# In[ ]:





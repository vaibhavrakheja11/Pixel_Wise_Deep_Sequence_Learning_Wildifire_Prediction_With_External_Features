#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, UpSampling2D
from keras import backend as K
import tensorflow as tf
import tarfile
import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, figure
import cv2
import logging
from keras.preprocessing.image import load_img


# In[5]:


logging.basicConfig(filename = '..\\test.log', level = logging.DEBUG)

f = open('test.log', mode = 'w') 


# In[6]:


os.getcwd()


# In[ ]:





# In[7]:


#initialize wandb and download dataset

hyperparams = {"num_epochs": 10, 
          "batch_size": 2,
          "height": 375,
          "width": 375}

config=hyperparams
#wandb.init(config=hyperparams)
#config = wandb.config

# print('uploading Test data')
# tar=tarfile.open('Simulated_Data/test.tar.gz')
# test=tar.extractall()

# print('uploading Train data')
# tar=tarfile.open('Simulated_Data/train.tar.gz')
# train=tar.extractall()



print('VAL_DIR')    
val_dir = 'Data/test_data'
train_dir = 'Data/train_data'
test_dir = 'Data/realtest'


# automatically get the data if it doesn't exist
#if not os.path.exists("catz"):
#    print("Downloading catz dataset...")
#    subprocess.check_output(
#        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)


# In[8]:


img = load_img(train_dir + '/cat_112j/cat_0.jpg')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)


# In[9]:


# generator to loop over train and test images


def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config['width'], config['height'], 3 * 5))
        output_images = np.zeros((batch_size, config['width'], config['height'], 3))
#         random.shuffle(cat_dirs)
        if (counter+batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img).convert('RGB') for img in sorted(input_imgs)]
            
            #print(sorted(input_imgs))
#             print(img , " : " , input_images[i].shape, 'ImAGE NAME :-', sorted(input_images))
            
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg").convert('RGB'))
            input_images[i] /= 255.
            output_images[i] /= 255.
        yield (input_images, output_images)
        counter += batch_size
        
steps_per_epoch = len(glob.glob(train_dir + "/*")) // config['batch_size']
validation_steps = len(glob.glob(val_dir + "/*")) // config['batch_size']


# In[10]:


len(glob.glob(train_dir + "/*"))//config['batch_size']


# In[11]:


len(glob.glob(val_dir + "/*")) // config['batch_size']


# In[12]:


#callback to log the images

class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(5, val_dir))
        output = self.model.predict(validation_X)
        print(output.shape)
#         wandb.log({
#             "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=2), axis=1)) for c in validation_X],
#             "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
#         }, commit=False)


# In[13]:


#Test the generator
gen = my_generator(2, train_dir)
input, output = next(gen)


videos, next_frame = next(gen)
print(videos[0].shape)
print(next_frame[0].shape)

figure()
imshow(videos[0][:,:,0:3])
figure()
imshow(videos[0][:,:,3:6])
figure()
imshow(videos[0][:,:,6:9])
figure()
imshow(videos[0][:,:,9:12])
figure()
imshow(videos[0][:,:,12:15])

figure()
imshow(next_frame[0][:,:,0:3])

# gen = my_generator(2, train_dir)
# videos, next_frame = next(gen)
# print(videos[0].shape)
# print(next_frame[0].shape)

# figure()
# imshow(videos[0][:,:,0:4])
# figure()
# imshow(videos[0][:,:,4:8])
# figure()
# imshow(videos[0][:,:,8:12])
# figure()
# imshow(videos[0][:,:,12:16])
# figure()
# imshow(videos[0][:,:,16:20])

# figure()
# imshow(next_frame[0][:,:,0:4])


# In[14]:


# # Function for measuring how similar two images are
# def perceptual_distance(y_true, y_pred):
#     y_true *= 255.
#     y_pred *= 255.
#     rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
#     r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
#     g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
#     b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    
    

#     return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


# In[15]:


# Function for measuring how similar two images are
def perceptual_distance(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    
    

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


# In[16]:


# wandb.init(config=hyperparams)
# config = wandb.config
config=hyperparams

model = Sequential()
model.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(config['height'], config['width'], 5 * 3)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same')),

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[17]:


val_dir


# In[18]:


# Baseline model - just return the last layer

from keras.layers import Lambda, Reshape, Permute

def slice(x):
    return x[:,:,:,:, -1]

config=hyperparams

model=Sequential()
model.add(Reshape((96,96,5,3), input_shape=(config['height'], config['width'], 5 * 3)))
model.add(Permute((1,2,4,3)))
model.add(Lambda(slice, input_shape=(96,96,3,5), output_shape=(96,96,3)))

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch//4,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps//4,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[ ]:


with open('..\\test.log') as f:
    print(f)


# In[ ]:



f.close()



# In[ ]:





# In[21]:


#for testing utk

cat_dirs = glob.glob(test_dir + "/*")
for each_dir in cat_dirs:
    input_images = np.zeros(
            (1, config['width'], config['height'], 3 * 5))
    output_images = np.zeros((1, config['width'], config['height'], 3))

    input_imgs = glob.glob(each_dir + "/cat_[0-5]*")
    imgs = [Image.open(img).convert('RGB') for img in sorted(input_imgs)]

    input_images[0] = np.concatenate(imgs, axis=2)
    output_images[0] = np.array(Image.open(
        each_dir + "/cat_result.jpg").convert('RGB'))
    input_images[0] /= 255.
    output_images[0] /= 255.    

    output = model.predict(input_images)
    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    


# In[ ]:


import cv2
import matplotlib.pyplot as plt
test_gen = test_generator(1, test_dir)
# test_img_set, output_img = next(test_gen)
# output = model.predict(test_img_set)

for test_imgs, output_imgs in test_gen:
    output = model.predict(test_imgs)
    a = perceptual_distance(output_img, output)
    print("Perceptual Distance :-", K.get_value(a))


# In[ ]:





# In[ ]:





# In[2]:





# In[ ]:





# In[1]:


a = perceptual_distance(output_img, output)
K.get_value(a)


# In[ ]:





# In[ ]:


output.shape


# In[ ]:


type(output_img)


# In[ ]:


type(output)


# In[3]:


# Baseline model - just return the last layer

from keras.layers import Lambda, Reshape, Permute

def slice(x):
    return x[:,:,:,:, -1]

config=hyperparams

model=Sequential()
model.add(Reshape((375,375,5,3), input_shape=(config['height'], config['width'], 5 * 3)))
model.add(Permute((1,2,4,3)))
model.add(Lambda(slice, input_shape=(375,375,3,5), output_shape=(375,375,3)))

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch//4,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps//4,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[ ]:





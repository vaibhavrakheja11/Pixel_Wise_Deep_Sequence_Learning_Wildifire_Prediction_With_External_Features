#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import matplotlib.pyplot as plt
import logging
from keras.preprocessing.image import load_img


# In[3]:


from PIL import Image
im = Image.open('Data/train_data/Fire_12_jd_181.tif')
im.show()


# In[93]:


t = np.array(Image.open('Data/train_data/Fire_12_jd_181.tif').getdata()).reshape([128,128])


# In[94]:


#d= t.reshape([128,128])
#im = Image.fromarray(d, mode='F') # float32
#im.save("Fire_12_jd_181.tiff", "TIFF")
plt.imshow(t, cmap="gray")
plt.show()


# In[90]:


t.shape


# In[91]:


# k = t.reshape((128,128,1))
# im = Image.fromarray(k, mode='F') 
# im.save("test3.tiff", "TIFF")
# k.shape


# In[7]:


t = np.concatenate([k,k,k,k,k,k,k],axis=-1)
t.shape


# In[33]:


cat_dirs = glob.glob(train_dir + "/*")
counter = 0
# if (counter+2 >= len(cat_dirs)):
#     counter = 0
print("kjo")
for i in range(1):
    input_imgs = glob.glob(cat_dirs[0 + i] + "/Fire_*")
    print("hh")
    print(input_imgs[0])
    for f in input_imgs:
        print(f)
        print("---xx---xx---xx---xx---")
#files = [f for f in glob.glob(cat_dirs + "/Fire_*.tif", recursive=True)]


# In[8]:


#initialise params
hyperparams = {"num_epochs": 10, 
          "batch_size": 2,
          "height": 480,
          "width": 480}

config=hyperparams


# In[9]:


#load data
val_dir = 'Data/test_data_div'
train_dir = 'Data/train_data_div'
test_dir = 'Data/eval_data_div'


# In[10]:


#check image format
img = load_img(train_dir + '/fire_01/Fire_12_jd_174.png')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)


# In[110]:


#define generator 
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
            input_imgs = glob.glob(cat_dirs[counter + i] + "/Fire_*")
            imgs = [Image.open(img).convert('RGB') for img in sorted(input_imgs)]
            
            #print(sorted(input_imgs))
#             print(img , " : " , input_images[i].shape, 'ImAGE NAME :-', sorted(input_images))
            
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/fire_result.tiff").convert('RGB'))
            input_images[i] /= 255.
            output_images[i] /= 255.
        yield (input_images, output_images)
        counter += batch_size
        
steps_per_epoch = len(glob.glob(train_dir + "/*")) // config['batch_size']
validation_steps = len(glob.glob(val_dir + "/*")) // config['batch_size']


# In[111]:


#len(glob.glob(train_dir + "/*"))//config['batch_size']


# In[112]:


#len(glob.glob(val_dir + "/*")) // config['batch_size']


# In[113]:


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


# In[114]:


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


# In[115]:


# # Function for measuring how similar two images are
# def perceptual_distance(y_true, y_pred):
#     y_true *= 255.
#     y_pred *= 255.
#     rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
#     r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
#     g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
#     b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    
    

#     return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


# In[116]:


# Function for measuring how similar two images are uttu
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


# In[117]:


# simple conv2D model 2 layers
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


# In[ ]:





# In[129]:


#local testing block uttu
cat_dirs = glob.glob(test_dir + "/*")

fig, ax = plt.subplots(len(cat_dirs), 2 ,figsize=(45,45))
print(len(cat_dirs))
for i, each_dir in enumerate(cat_dirs):
    input_images = np.zeros(
            (1, config['width'], config['height'], 3 * 5))
    output_images = np.zeros((1, config['width'], config['height'], 3))

    #input_imgs = glob.glob(each_dir + "/cat_[0-5]*")
    input_imgs = glob.glob(each_dir + "/Fire_*")
    imgs = [Image.open(img).convert('RGB') for img in sorted(input_imgs)]
    print(input_imgs)
    input_images[0] = np.concatenate(imgs, axis=2)
    output_images[0] = np.array(Image.open(
        each_dir + "/fire_result.png").convert('RGB'))
    input_images[0] /= 255.
    output_images[0] /= 255.    

    output_pred = model.predict(input_images)
    
    ax[i, 0].imshow(output_images[0])
    ax[i, 1].imshow(output_pred[0])

    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    
    
plt.show()


# In[ ]:


# test_gen = test_generator(1, test_dir)
# # test_img_set, output_img = next(test_gen)
# # output = model.predict(test_img_set)

# for test_imgs, output_imgs in test_gen:
#     output = model.predict(test_imgs)
#     a = perceptual_distance(output_img, output)
#     print("Perceptual Distance :-", K.get_value(a))


# In[ ]:


#plt.imshow(output[0])
#plt.imshow(output_img[0])


# In[ ]:


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
                    steps_per_epoch=steps_per_epoch,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[ ]:


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


#checking the perpetual distance of the predictions made by Baseline model-1

cat_dirs = glob.glob(test_dir + "/*")

fig, ax = plt.subplots(len(cat_dirs), len(cat_dirs), )

for i, each_dir in enumerate(cat_dirs):
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

    output_pred = model.predict(input_images)
    
    ax[i, 0].imshow(output_images[0])
    ax[i, 1].imshow(output_pred[0])

    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    
    
plt.show()


# In[ ]:


#Just return the last layer, functional style

from keras.layers import Lambda, Reshape, Permute, Input
from keras.models import Model

def slice(x):
    return x[:,:,:,:, -1]

config=hyperparams


inp = Input((config['height'], config['width'], 5 * 3))
reshaped = Reshape((375,375,5,3))(inp)
permuted = Permute((1,2,4,3))(reshaped)
last_layer = Lambda(slice, input_shape=(375,375,3,5), output_shape=(375,375,3))(permuted)
model=Model(inputs=[inp], outputs=[last_layer])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[ ]:


#checking the perpetual distance of the predictions made by functional model-1

cat_dirs = glob.glob(test_dir + "/*")

fig, ax = plt.subplots(len(cat_dirs), len(cat_dirs), )

for i, each_dir in enumerate(cat_dirs):
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

    output_pred = model.predict(input_images)
    
    ax[i, 0].imshow(output_images[0])
    ax[i, 1].imshow(output_pred[0])

    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    
    
plt.show()


# In[ ]:


# Conv3D

from keras.layers import Lambda, Reshape, Permute, Input, add, Conv3D
from keras.models import Model

def slice(x):
    return x[:,:,:,:, -1]

hyperparams["num_epochs"] = 20
config=hyperparams


inp = Input((config['height'], config['width'], 5 * 3))
reshaped = Reshape((375,375,5,3))(inp)
permuted = Permute((1,2,4,3))(reshaped)
last_layer = Lambda(slice, input_shape=(375,375,3,5), output_shape=(375,375,3))(permuted)
conv_output = Conv3D(1, (3,3,3), padding="same")(permuted)
conv_output_reshape = Reshape((375,375,3))(conv_output)
combined = add([last_layer, conv_output_reshape])

model=Model(inputs=[inp], outputs=[combined])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[130]:


# Conv2DLSTM with Gaussian Noise

from keras.layers import Lambda, Reshape, Permute, Input, add, Conv3D, GaussianNoise, ConvLSTM2D
from keras.models import Model

def slice(x):
    return x[:,:,:,:, -1]

config=hyperparams


inp = Input((config['height'], config['width'], 5 * 3))
reshaped = Reshape((480,480,5,3))(inp)
permuted = Permute((1,2,4,3))(reshaped)
noise = GaussianNoise(0.1)(permuted)
last_layer = Lambda(slice, input_shape=(480,480,3,5), output_shape=(480,480,3))(noise)
permuted_2 = Permute((4,1,2,3))(noise)

conv_lstm_output_1 = ConvLSTM2D(6, (3,3), padding='same')(permuted_2)
conv_output = Conv2D(3, (3,3), padding="same")(conv_lstm_output_1)
combined = add([last_layer, conv_output])

model=Model(inputs=[inp], outputs=[combined])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch//4,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps//4,
    validation_data=my_generator(config['batch_size'], val_dir))



# In[131]:


cat_dirs = glob.glob(test_dir + "/*")

fig, ax = plt.subplots(len(cat_dirs), 2 ,figsize=(45,45))
print(len(cat_dirs))
for i, each_dir in enumerate(cat_dirs):
    input_images = np.zeros(
            (1, config['width'], config['height'], 3 * 5))
    output_images = np.zeros((1, config['width'], config['height'], 3))

    #input_imgs = glob.glob(each_dir + "/cat_[0-5]*")
    input_imgs = glob.glob(each_dir + "/Fire_*")
    imgs = [Image.open(img).convert('RGB') for img in sorted(input_imgs)]
    print(input_imgs)
    input_images[0] = np.concatenate(imgs, axis=2)
    output_images[0] = np.array(Image.open(
        each_dir + "/fire_result.png").convert('RGB'))
    input_images[0] /= 255.
    output_images[0] /= 255.    

    output_pred = model.predict(input_images)
    
    ax[i, 0].imshow(output_images[0])
    ax[i, 1].imshow(output_pred[0])

    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    
    
plt.show()


# In[1]:



from keras.layers import Lambda, Reshape, Permute, Input, add, Conv3D, GaussianNoise, concatenate
from keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, Add
from keras.models import Model

def slice(x):
    return x[:,:,:,:, -1]

config=hyperparams


c=4

inp = Input((config['height'], config['width'], 5 * 7))
reshaped = Reshape((128,128,5,7))(inp)
permuted = Permute((1,2,4,3))(reshaped)
noise = GaussianNoise(0.1)(permuted)
last_layer = Lambda(slice, input_shape=(128,128,7,5), output_shape=(128,128,7))(noise)
x = Permute((4,1,2,3))(noise)
x =(ConvLSTM2D(filters=c, kernel_size=(3,3),padding='same',name='conv_lstm1', return_sequences=True))(x)


c1=(BatchNormalization())(x)
x = Dropout(0.2)(x)
x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c1)

x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm3',return_sequences=True))(x)
c2=(BatchNormalization())(x)
print(c2.shape)
x = Dropout(0.2)(x)

x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c2)
x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm4',return_sequences=True))(x)

x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm5',return_sequences=True))(x)
x =(BatchNormalization())(x)
print(x.shape)

x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm6',return_sequences=True))(x)
x =(BatchNormalization())(x)
print(x.shape)
x = Add()([c2, x])
x = Dropout(0.2)(x)

x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
x =(ConvLSTM2D(filters=c,kernel_size=(3,3),padding='same',name='conv_lstm7',return_sequences=False))(x)
x =(BatchNormalization())(x)
combined = concatenate([last_layer, x])
combined = Conv2D(3, (1,1))(combined)
model=Model(inputs=[inp], outputs=[combined])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch//4,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps//4,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[ ]:





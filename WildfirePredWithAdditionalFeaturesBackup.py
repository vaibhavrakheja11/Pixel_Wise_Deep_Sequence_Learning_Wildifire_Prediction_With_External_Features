#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from PIL import Image
im = Image.open('Data/train_data/Fire_12_jd_174.tif')
np.array(im).shape


# In[3]:


t = np.array(Image.open('Data/train_data/Fire_12_jd_174.tif').getdata()).reshape([128,128])


# In[4]:


# d= t.reshape([128,128])
# print(np.max(d))
# im = Image.fromarray(d, mode='F') # float32
# im.save("test4.tiff", "TIFF")
plt.imshow(t,"gray")
plt.show()


# In[5]:


k = t.reshape((128,128,1))
k.shape
#k.show()
# t = np.concatenate([k,k,k,k,k,k,k],axis=-1)
# t.shape


# In[6]:


#load data
test_dir = 'data_f/test_div'
train_dir = 'data_f/train_div'
val_dir = 'data_f/eval_div'


# In[7]:


# #reshape train data

# cat_dirs = glob.glob(train_dir + "/*")
# counter = 0
# # if (counter+2 >= len(cat_dirs)):
# #     counter = 0
# cpt = sum([len(files) for w, r, files in os.walk("/Users/vaibhav/Documents/University of Alberta MM/Term-1/MM-811/Project/Keras Convo-LSTM New Data/Data_tiff/train_div")])
# len1 = cpt/6
# #print(len1)
# for i in range(int(len1)):
#     input_imgs = glob.glob(cat_dirs[0 + i] + "/Fire_*")
#     #print(input_imgs[0])
#     for f in input_imgs:
#         #print(f)
#         print("---xx---xx---xx---xx---")
#         print(f[20:52])#image name
#         #print(f[0:34]) #image path
#         im = Image.open(f)
        
#         #convert tif to Tiff
#         t = np.array(im.getdata())
#         #d= t.reshape((128,128))
#         k = t.reshape((128,128))
#         #k.shape
#         #t = np.concatenate([k,k,k,k,k,k,k],axis=-1)
#         #print(t.shape)
#         im = Image.fromarray(k, mode='F') # float32
#         im.save(f+'f' , "TIFF")


# In[8]:


# im = Image.open('Data_TIFF/train_div/Data_slice_30/Fire_27_jd_188.tiff')
# #im.show()
# t = np.array(im.getdata())
# t= t.reshape([128,128])
# plt.imshow(t, cmap="gray")
# plt.show()


# In[9]:


# #reshape test data

# cat_dirs = glob.glob(test_dir + "/*")
# counter = 0
# # if (counter+2 >= len(cat_dirs)):
# #     counter = 0
# cpt = sum([len(files) for w, r, files in os.walk("/Users/vaibhav/Documents/University of Alberta MM/Term-1/MM-811/Project/Keras Convo-LSTM New Data/Data_tiff/test_div")])
# len1 = cpt/6
# #print(len1)
# for i in range(int(len1)):
#     input_imgs = glob.glob(cat_dirs[0 + i] + "/Fire_*")
#     #print(input_imgs[0])
#     for f in input_imgs:
#         #print(f)
#         print("---xx---xx---xx---xx---")
#         #print(f[34:52])#image name
#         print(f[0:34]) #image path
#         im = Image.open(f)
        
#         #convert tif to Tiff
#         t = np.array(im.getdata())
#         #d= t.reshape((128,128))
#         k = t.reshape((128,128))
#         #k.shape
#         #t = np.concatenate([k,k,k,k,k,k,k],axis=-1)
#         #print(t.shape)
#         im = Image.fromarray(k, mode='F') # float32
#         im.save(f+'f', "TIFF")


# In[10]:


# #reshape eval data

# cat_dirs = glob.glob(val_dir + "/*")
# counter = 0
# # if (counter+2 >= len(cat_dirs)):
# #     counter = 0
# #copy the path where the slice_data folders are kept
# cpt = sum([len(files) for w, r, files in os.walk("/Users/vaibhav/Documents/University of Alberta MM/Term-1/MM-811/Project/Keras Convo-LSTM New Data/Data_tiff/eval_div")]) 

# #total number of folder = number of files in directory/6(6 images per folder)
# len1 = cpt/6

# #print(len1)
# for i in range(int(len1)):
#     input_imgs = glob.glob(cat_dirs[0 + i] + "/Fire_*")
#     #print(input_imgs[0])
#     for f in input_imgs:
#         #print(f)
#         print("---xx---xx---xx---xx---")
#         #print(f[34:52])#image name
#         print(f[0:34]) #image path
#         im = Image.open(f)
        
#         #convert tif to Tiff
#         t = np.array(im.getdata())
#         #d= t.reshape((128,128))
#         k = t.reshape((128,128))
#         #k.shape
#         #t = np.concatenate([k,k,k,k,k,k,k],axis=-1)
#         #print(t.shape)
#         im = Image.fromarray(k, mode='F') # float32
#         im.save(f, "TIFF")


# In[11]:


# img = Image.open(train_dir +'/Data_slice_30/Fire_27_jd_183.tif')
# t = np.array(img.getdata())
# #d= t.reshape([128,128,7])
# t.shape
# im.show()


# In[12]:


#check image format
img = load_img(train_dir + '/fire_01/Fire_12_jd_174.png')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)


# In[27]:


#initialise params
hyperparams = {"num_epochs": 100, 
          "batch_size": 2,
          "height": 128,
          "width": 128}

config=hyperparams


# In[28]:


pwd


# In[29]:


# from PIL import Image
# im = Image.open('data_f/Data_slice_1/Fire_12_jd_174.tif')
# #im.show()
# t = np.array(im.getdata())
# d= t.reshape([128,128])
# im = Image.fromarray(d, mode='F') # float32
# im.save("temp1_ws.tiff", "TIFF")
# im = Image.open('temp1_ws.tiff')
# #im.show()
# plt.imshow(d, cmap="gray")
# plt.show()
# #d.shape

# im = Image.open('data_f/Data_slice_1/Fire_12_jd_174_wind_speed.tif')
# #im.show()
# t = np.array(im.getdata())
# k= t.reshape([128,128])
# im = Image.fromarray(d, mode='F') # float32
# im.save("temp1_ws.tiff", "TIFF")
# im = Image.open('temp1_ws.tiff')
# #im.show()
# plt.imshow(k, cmap="gray")
# plt.show()
# #k.shape

# d=d.reshape([128,128,1])
# k=k.reshape([128,128,1])
# p = np.concatenate([d,k],axis=-1)
# print(p.shape)
# # plt.imshow(p, cmap="gray")
# # plt.show()

# # im = Image.fromarray(p, mode='F') # float32
# # im.save("temp1_concat.tiff", "TIFF")


# In[30]:


# im = Image.open('temp1_ws.tiff')
# #im.show()
# plt.imshow(d, cmap="gray")
# plt.show()
# t = np.array(im.getdata())
# t.shape
# imag = t.reshape([128,128,2])
# # imag.shape
# im = Image.open('Data_TIFF/test_div/Data_slice_62/Fire_35_jd_182.tiff')
# im.show()
# t = np.array(im.getdata())
# t= t.reshape([128,128])
# plt.imshow(t, cmap="gray")
# plt.show()


# In[31]:


#define generator
def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config['width'], config['height'], 7 * 5))
        output_images = np.zeros((batch_size, config['width'], config['height'], 1))
#         random.shuffle(cat_dirs)
        if (counter+batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/Fire_*")
            #print(input_imgs)
            
            parent_file_list = []
            for file in input_imgs:
                f = '_'.join(file.split('/')[-1].split('_')[0:4]).split('.')[0]
                if not f in parent_file_list:
                    parent_file_list.append(f)
            
            
            combined_list =[]
            print(parent_file_list)
            for each_file in parent_file_list:
                each_file_list = glob.glob(cat_dirs[counter + i] + "/"+each_file+"*")
                
                imgs = [np.array(Image.open(img).getdata()).reshape([128,128,1]) for img in sorted(each_file_list)]
                
                combined_list.append(np.concatenate(imgs, axis = 2))
                
                
                
                
                
            input_images[i] = np.concatenate(combined_list, axis=2)
                
            
#             #print(parent_file_list)
#             imgs = [np.array(Image.open(img).getdata()).reshape([128,128,1]) for img in sorted(input_imgs)]
            
#             #print(len(imgs))
#             #print(sorted(input_imgs))
#             for img in imgs:
#                 print(img , " : " , input_images[i].shape, 'ImAGE NAME :-', sorted(input_images))

#                 input_images[i] = np.concatenate(imgs, axis=2)

            output_images[i] = np.array(Image.open(
            cat_dirs[counter + i] + "/fire_result.tif").getdata()).reshape([128,128,1])
            input_images[i] /= 255.
            output_images[i] /= 255.
        print(input_images.shape)
        yield (input_images, output_images)
        counter += batch_size
        
steps_per_epoch = len(glob.glob(train_dir + "/*")) // config['batch_size']
validation_steps = len(glob.glob(val_dir + "/*")) // config['batch_size']




# In[24]:


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


# In[25]:


#Test the generator
gen = my_generator(2, train_dir)
input, output = next(gen)


videos, next_frame = next(gen)
print(videos[0].shape)
print(next_frame[0].shape)

figure()
plt.imshow(np.array(videos[0][:,:,0]).reshape([128,128]), cmap='gray')
figure()

plt.imshow(np.array(videos[0][:,:,7]).reshape([128,128]), cmap='gray')
figure()

plt.imshow(np.array(videos[0][:,:,14]).reshape([128,128]), cmap='gray')
figure()

plt.imshow(np.array(videos[0][:,:,21]).reshape([128,128]), cmap='gray')
figure()

plt.imshow(np.array(next_frame[0][:,:,0]).reshape([128,128]), cmap='gray')
# imshow(videos[0][:,:,7:8])
# figure()
# imshow(videos[0][:,:,14:15])
# figure()
# imshow(videos[0][:,:,21:22])
# figure()
# imshow(videos[0][:,:,28:29])

# figure()
# imshow(next_frame[0][:,:,0])
plt.show()


# In[26]:


def perceptual_distance(y_true, y_preThe d):
    y_pred = tf.convert_to_tensor(y_pred, np.float32) #replace with sigmoid and cross entropy.
    y_true = tf.convert_to_tensor(y_true, np.float32)
    
    rmean = (y_true[:, :, :, 0] - y_pred[:, :, :, 0]) / 2
    

    return K.mean(K.sqrt(rmean))




# In[20]:


# simple conv2D model 2 layers
config=hyperparams

model = Sequential()
model.add(Conv2D(1, (3, 3), activation='relu', padding='same', input_shape=(config['height'], config['width'], 5 * 7)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same')),

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps,
    validation_data=my_generator(config['batch_size'], test_dir))


# In[21]:


#local testing block uttu
cat_dirs = glob.glob(val_dir + "/*")
counter = 0

fig, ax = plt.subplots(len(cat_dirs), 2 ,figsize=(45,45))
print(len(cat_dirs))
for i, each_dir in enumerate(cat_dirs):
#     input_images = np.zeros(
#             (1, config['width'], config['height'], 7 * 5))
#     output_images = np.zeros((1, config['width'], config['height'], 1))

#     #input_imgs = glob.glob(each_dir + "/cat_[0-5]*")
#     input_imgs = glob.glob(each_dir + "/Fire_*")
#     imgs = [np.array(Image.open(img).getdata()).reshape([128,128,1]) for img in sorted(input_imgs)]
#     print(input_imgs)
#     input_images[0] = np.concatenate(imgs, axis=2)
#     output_images[0] = np.array(Image.open(
#         each_dir + "/fire_result.tif")).reshape([128,128,1])
    
    input_images = np.zeros(
            (2, config['width'], config['height'], 7 * 5))
    output_images = np.zeros((2, config['width'], config['height'], 1))
#         random.shuffle(cat_dirs)
    if (counter+2 >= len(cat_dirs)):
        counter = 0

    input_imgs = glob.glob(cat_dirs[counter + i] + "/Fire_*")
    #print(input_imgs) 
    parent_file_list = []
    for file in input_imgs:
        f = '_'.join(file.split('/')[-1].split('_')[0:4]).split('.')[0]
        if not f in parent_file_list:
            parent_file_list.append(f)

    combined_list =[]
    print(parent_file_list)
    for each_file in parent_file_list:
        each_file_list = glob.glob(cat_dirs[counter + i] + "/"+each_file+"*")

        imgs = [np.array(Image.open(img).getdata()).reshape([128,128,1]) for img in sorted(each_file_list)]

        combined_list.append(np.concatenate(imgs, axis = 2))

    input_images[0] = np.concatenate(combined_list, axis=2)

    output_images[0] = np.array(Image.open(
    cat_dirs[counter + i] + "/fire_result.tif").getdata()).reshape([128,128,1])
    
    
    
    
    #input_images[0] /= 255.
    #output_images[0] /= 255.    

    output_pred = model.predict(input_images)
    
    ax[i, 0].imshow(np.array(output_images[0]).reshape(128,128), cmap="gray")
    ax[i, 1].imshow(np.array(output_pred[0]).reshape(128,128), cmap="gray")
    #ax[i, 1].imshow(output_pred[0], cmap="gray")
    #ax[i, 1].imshow(output_pred[0])

    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    
    
plt.show()


# In[32]:


#ConvoLstm2d with gaussian noise
from keras.layers import Lambda, Reshape, Permute, Input, add, Conv3D, GaussianNoise, ConvLSTM2D
from keras.models import Model

def slice(x):
    return x[:,:,:,:, -1]

config=hyperparams


inp = Input((config['height'], config['width'], 5 * 7))
reshaped = Reshape((128,128,5,7))(inp)
permuted = Permute((1,2,4,3))(reshaped)
noise = GaussianNoise(0.1)(permuted)
last_layer = Lambda(slice, input_shape=(128,128,7,5), output_shape=(128,128,1))(noise)
permuted_2 = Permute((4,1,2,3))(noise)

conv_lstm_output_1 = ConvLSTM2D(6, (3,3), padding='same')(permuted_2)
conv_output = Conv2D(1, (3,3), padding="same")(conv_lstm_output_1)
combined = add([last_layer, conv_output])

model=Model(inputs=[inp], outputs=[combined])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config['batch_size'], train_dir),
                    steps_per_epoch=steps_per_epoch,
                    epochs=config['num_epochs'], callbacks=[
    ImageCallback()],
    validation_steps=validation_steps,
    validation_data=my_generator(config['batch_size'], val_dir))


# In[33]:


#local testing block uttu
cat_dirs = glob.glob(val_dir + "/*")
counter = 0

fig, ax = plt.subplots(len(cat_dirs), 2 ,figsize=(45,45))
print(len(cat_dirs))
for i, each_dir in enumerate(cat_dirs):
#     input_images = np.zeros(
#             (1, config['width'], config['height'], 7 * 5))
#     output_images = np.zeros((1, config['width'], config['height'], 1))

#     #input_imgs = glob.glob(each_dir + "/cat_[0-5]*")
#     input_imgs = glob.glob(each_dir + "/Fire_*")
#     imgs = [np.array(Image.open(img).getdata()).reshape([128,128,1]) for img in sorted(input_imgs)]
#     print(input_imgs)
#     input_images[0] = np.concatenate(imgs, axis=2)
#     output_images[0] = np.array(Image.open(
#         each_dir + "/fire_result.tif")).reshape([128,128,1])
    
    input_images = np.zeros(
            (2, config['width'], config['height'], 7 * 5))
    output_images = np.zeros((2, config['width'], config['height'], 1))
#         random.shuffle(cat_dirs)
    if (counter+2 >= len(cat_dirs)):
        counter = 0

    input_imgs = glob.glob(cat_dirs[counter + i] + "/Fire_*")
    #print(input_imgs) 
    parent_file_list = []
    for file in input_imgs:
        f = '_'.join(file.split('/')[-1].split('_')[0:4]).split('.')[0]
        if not f in parent_file_list:
            parent_file_list.append(f)

    combined_list =[]
    print(parent_file_list)
    for each_file in parent_file_list:
        each_file_list = glob.glob(cat_dirs[counter + i] + "/"+each_file+"*")

        imgs = [np.array(Image.open(img).getdata()).reshape([128,128,1]) for img in sorted(each_file_list)]

        combined_list.append(np.concatenate(imgs, axis = 2))

    input_images[0] = np.concatenate(combined_list, axis=2)

    output_images[0] = np.array(Image.open(
    cat_dirs[counter + i] + "/fire_result.tif").getdata()).reshape([128,128,1])
    
    
    
    
    #input_images[0] /= 255.
    #output_images[0] /= 255.    

    output_pred = model.predict(input_images)
    print(output_pred.shape)
    
    ax[i, 0].imshow(np.array(output_images[0]).reshape(128,128), cmap="gray")
    im1 = np.array(output_pred[0]).reshape(128,128,7)
    print(im1[:,:,0].shape)
#     plt.imshow(im[:,:,0])
    ax[i, 1].imshow(im1[:,:,0], cmap="gray")
    #ax[i, 1].imshow(output_pred[0], cmap="gray")
    #ax[i, 1].imshow(output_pred[0])

    a = perceptual_distance(output_images, output)
    print("Perceptual Distance :-", K.get_value(a))
    
    
plt.show()


# In[ ]:





# In[ ]:





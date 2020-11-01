## Deep Sequence learning for wildfire spread prediction

## Project Introduction

In North America, wildfires occur frequently and have direct impacts on society, economy, and environment. In Canada, wildfire management agencies spend 800 million dollars each year on wildfire management, which is projected to increase due to the human-introduced climate change. Therefore, improving the efficiency of the fire management decision-making process is one of the most important tasks for wildfire management agencies. In particular, wildfire spread prediction is one of the essential yet challenging tasks. 

Current methods in place for wildfire spread prediction is based either on empirical or physical models, which are all limited by the imperfect understanding of the physical processes and data quality. The recent advancement in pixel-wise deep sequence learning models may be a feasible solution for this problem. However, there are no existing deep learningbased works in the context of wildfire spread prediction, In addition, there is no existing standard (e.g., how to build training dataset and framework architecture) for using deep learning methods for wildfire spread prediction. 

As the first step of a complicated question, we propose to implement the pixel-wise deep learning methods that combine the convolutional–recurrent approaches for wildfire spread prediction. We will build the training dataset using the historical wildfire mapped from the satellite images. For evaluation, we will use our own benchmark dataset following the standard object segmentation evaluation methods.

Wildfire model generally use predicitive models to predict the fire growth, however, most of the models those are currently working have irregularities in predition. One of them being is to make a prediction with the image data only, ignoring the environment factors like wind speed, wind direction, fuel type, moisture, precipitation etc. While these factors largly affect the progression of wildfire, this models attempts to use these environmental factors as external features effecting the wildfire. 




## Data
Simulated Data :- http://www.shodor.org/interactivate/activities/FireAssessment/

### Division of Data

The input training images are the daily (mid-night to mid-night) area burned images for a particular wildfire. For example, if a wildfire burned for 6 days, we will have 6 training images for this wildfire from day 1 to day 6.
The goal is to train the model to learn the pattern of wildfire images for first 5 days, and the model can predict the fire
image at day 6. To simplify the training task, we categorized the training images, where 0 represents unburned area and 1 represents burned area. The feature variables (e.g, weather and fuel moisture) are also prepared as a binary image and the values in the training images are normalized to 0-1. Details of each data preparation step are described as follows.

------

## Getting started

To start trainning with real wildfire data.

Open Jupyter Notebook from terminal (or any other method you preferred)

```cmd
$ jupyter notebook
```
There are a few cases in which the problem statement is divided.
We have divided the the problem statement into 4 problem that will be solved chronologically. 
- N + 1th frame prediction with Simulated Data Models. `SampleData`
- N + 1th frame prediction with external features using simple CNN model. `tiffworksheet`
- N + 1th frame prediction with external features using ConvLSTM2D model. `WildfirePredWithAdditionalFeatures`


If you want to try model without any additional features, please open file  [SampleData.py] (http://localhost:8888/notebooks/SampleData.ipynb).
If you want to try model without any additional features, please open file  [tiffworksheet.py](http://localhost:8888/notebooks/tiffworksheet.ipynb).
 If you want to work on the model with additional features that predicts n + 1th day image, please run [WildfirePredWithAdditionalFeatures.py](http://localhost:8888/notebooks/WildfirePredWithAdditionalFeatures.ipynb) contain supplementary features.

If you have opend the ipynb(jupyter notebook), run each section in order, you may see some examples and stardard images at the beginning. After running the section start with `#Conv2DLSTM with Gaussian Noise` ( in tiffworksheet.ipynb) or the section start with `#ConvoLstm2D with gaussian noise`( in WildfirePredWithAdditionalFeatures.ipynb), the corresponding model start training with real fire data-set which may cost some time depending upon the gpu speeds.


Also you can customize the parameters of model in following section:

```python
#initialise params
hyperparams = {"num_epochs": 10, 
          "batch_size": 2,
          "height": 128,
          "width": 128}

config=hyperparams
```

After trainning, the last block in the file (start with `#local testing block` ) will give you the test images and evaluation results. There will be two columns of results, ground-truth is on the left and the right side is the prediction for the n+1th frame of fire.


### - Known Errors
If you get problem as the following: `$ UnicodeDecodeError: 'ascii' codec can't decode byte 0xe5 in position 4: ordinal not in range(128)`, then try:

```python
$ LANG=zn jupyter notebook
```

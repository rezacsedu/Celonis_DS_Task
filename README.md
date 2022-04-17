# Gesture detection from time-series data and building production ready app
From a given time-series data, the task is to lassify it into one of the 8 classes. 


## Task 1: Gesture detection from time-series data 

  - A. We want you to use Python and numpy for all aspects of the "data science workflow" data preprocessing/feature extraction/ML-training/ML_test". (a few exceptions below)
  - B. Please implement and train a logistic regression model by hand (plain numpy)
  - C. If you want to use other ML-models (neural networks, support vector machines), feel free to use existing libraries. (Please, do not invest too much time here)
  - D. You can (should) use libraries for visualizing the results. (whatever you feel makes sense to visualize)

## Dataset
Downloaded the dataset from this [link](zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip). THis dataset is part of the paper "uWave: Accelerometer-based personalized gesture recognition and its applications" by Jiayang Liu et al. (see more at https://www.yecl.org/publications/liu09percom.pdf). Unpacking the data, leaves several .rar files, with the following meaning:

● Each .rar file includes the gesture samples collected from one user on one day. The .rar files are named as U$userIndex ($dayIndex).rar, where $userIndex is the index of the participant from 1 to 8, and $dayIndex is the index of the day from 1 to 7.
● Inside each .rar file, there are .txt files recording the time series of acceleration of each gesture. The .txt files are named as [somePrefix]$gestureIndex-$repeatIndex.txt, where $gestureIndex is the index of the gesture as in the 8-gesture vocabulary, and $repeatIndex is the index of the repetition of the same gesture pattern from 1 to 10.
● In each .txt file, the first column is the x-axis acceleration, the second y-axis acceleration, and the third z-axis acceleration. The unit of the acceleration data is G, or acceleration of gravity. 

Besides, I did some manual way of resturucturing the folders for gestures, gesture 1, gesture 2, ...., gesture 8, etc. My bad, perhaps that could be done in a automated way too. But anyway, sometimes we do things in a brute-force way so ;) Even I wrote another script for data download and extraction of .rar files. Please see naive_data_prep.py for more detail. 

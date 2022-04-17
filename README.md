## Gesture detection from time-series data and building production ready app
From a given time-series data, the task is to lassify it into one of the 8 classes. 


### Task 1: Gesture detection from time-series data 

  - T1.1: Use Python and numpy for all aspects of the "data science workflow" data preprocessing/feature extraction/ML-training/ML_test" (a few exceptions below)
  - T1.2: Please implement and train a logistic regression model by hand (plain numpy)
  - T1.3: If you want to use other ML-models (neural networks, support vector machines), feel free to use existing libraries (please, do not invest too much time here)
  - T1.4: You can/should use libraries for visualizing the results (whatever you feel makes sense to visualize).

### Dataset
Downloaded the dataset from this [link](zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip). This dataset is part of the paper "uWave: Accelerometer-based personalized gesture recognition and its applications" by Jiayang Liu et al. (see more at https://www.yecl.org/publications/liu09percom.pdf). Unpacking the data, leaves several .rar files, with the following meaning:

  - Each .rar file includes the gesture samples collected from one user on one day. The .rar files are named as U$userIndex ($dayIndex).rar, where $userIndex is the index of the participant from 1 to 8, and $dayIndex is the index of the day from 1 to 7.
  - Inside each .rar file, there are .txt files recording the time series of acceleration of each gesture. The .txt files are named as [somePrefix]$gestureIndex-$repeatIndex.txt, where $gestureIndex is the index of the gesture as in the 8-gesture vocabulary, and $repeatIndex is the index of the repetition of the same gesture pattern from 1 to 10.
  - In each .txt file, the first column is the x-axis acceleration, the second y-axis acceleration, and the third z-axis acceleration. The unit of the acceleration data is G (i.e., acceleration of gravity). 

<p align="center"><img src="imgs/gesture.png?" width="400" height="300"></p>

In the above picture, 8 different types of gestures have been shown, where the dot denotes the start and the arrow the end of the gesture. 

The dataset had some unwanted files too, e.g., 1.txt containing information such as "4-9: precision(11), sample frequency(100)". However, according to above description, such files are unnecessary. Therefore, I removed them keeping all the files named "[somePrefix]repeatIndex.txt". Besides, I had to do some manual way of resturucturing the folders for gestures, gesture 1, ...., gesture 8, etc. My bad, perhaps that could be done in a automated way too. Anyway, sometimes we do things in a brute-force way! Even, I wrote another script for data download and extraction of .rar files. See [naive_data_prep.py]((https://github.com/rezacsedu/Celonis_DS_Task/blob/main/utils/naive_data_prep.py)) for more detail. After restructuring and removing unnecessary files, I created another .zip file where folders are stuructured as follows: 

<p align="center"><img src="imgs/struct.png?" width="600" height="300"></p>

### Task 2: How to make a product ready ML software?
As just training and evaluating a model is of no use unless it can be used as a web application, e.g., own website, cloud, or production ready environment. Therefore, we need to think how a model and its associated workflow/pipeline can be converted as a ML software product. 

  - How would you design a devops pipeline using e.g. Github Actions for a Python package? Which functionalities would you include to ensure code quality and consistency?
  - Assuming the pipeline you implemented will be deployed as a product. Now the customer also wants to enable real time classification and consume an API that returns
the classification results. How would you fit that into the existing architecture?
  - The whole system has been a huge success and also other customers want to use it. How would you adapt everything to be able to serve multiple customers  ==, especially keep in mind scalability and data privacy.
  - What would you recommend to automatically transfer machine learning models to production by running microservices for inferencing?

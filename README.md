## Gesture detection from time-series data and building production ready app
From a given time-series data, the task is to lassify it into one of the 8 classes. There are two aspects of the challenge: model building (from scratch and using libraries) and making the model as a ML software to be deployed in production environment. 

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

<p align="center"><img src="imgs/gesture.png?" width="375" height="275"></p>

In the above picture, 8 different types of gestures have been shown, where the dot denotes the start and the arrow the end of the gesture. 

#### Cleaning and restructuring
The dataset had some unwanted files too, e.g., 1.txt containing information such as "4-9: precision(11), sample frequency(100)". However, according to above description, such files are unnecessary. Therefore, I removed them keeping all the files named "[somePrefix]repeatIndex.txt". Besides, I had to do some manual way of resturucturing the folders for gestures, gesture 1, ...., gesture 8, etc. My bad, perhaps that could be done in a automated way too. Anyway, sometimes we do things in a brute-force way! Even, I wrote another script for data download and extraction of .rar files. See [naive_data_prep.py]((https://github.com/rezacsedu/Celonis_DS_Task/blob/main/utils/naive_data_prep.py)) for more detail. After restructuring and removing unnecessary files, I created another .zip file where folders are stuructured as follows: 

<p align="center"><img src="imgs/struct.png?" width="500" height="230"></p>

### Solution
First, I implemented and trained a baseline logistic regression model on the feature extracted, which I evealuated on the test set w.r.t different performance metrics, confusion metric, and classification report. Then we compare it wioth sklearn-based logistic regression and a well-studied TimeSeriesForestClassifier from the SKTIME library. I'm really not expecting it to perform great because logistic regression is idelally not suitable for time series data. 

#### Task 1.1: Feature extraction
To extract feature, I wrote a script named data_util.py. THere are three methods in this script: 
  - **extract_gesture**: it taskes the path of respective user and returns a list of acceleration across all x-, y-, and z-axes. 
  - **create_dataset**: it taskes a set of gestures as input and generates a time series and labels. More technically, it creates a time series data by stacking the samples of all the gestures (total 4,480 samples) side by side, such that: i) the dimension of each sample is (n_samples, 3 * n_features) = (4480, 942), where the number of features = 314 * 3 = 942 and the nmber of n_samples = len(gesture) * 8 = 560 * 8 = 4480.  
  - **getFeatureSKTime**: this function is used to extract features using the sktime library. It takes the paths of training and set .ts file and returns the array of features and labels (X, y).

#### Task 1.2: Implementing logistic regression
A simple baseline logistic regression has been implemented in Numpy (without any regularizers) in the [lr.py script](https://github.com/rezacsedu/Celonis_DS_Task/blob/main/lr.py). The algorithm has several components: 

1. **An activation function**: based on which the predictions will be made. We take z - product of features with weights and generate discrete class over the classes. In fact, this makes the logistic regression different from linear regression.
2. **Cost function**: Cross-entropy between two probability distributions of y and p_pred
3. **Random initialization of parameters**: we just randomly initialize parameters 'w' and 'b' with 0 to start with (w vector has the shape (942, 1)), whereas the values will be updated using the optimization function. Then the cross-entropy cost function is optimze to minimize the cost. 
4. **Forward propagation and optimization**: Using the gradients descent algorithm, the gradients of the 'w' and 'b' parameters are computed and the parameters are updated, as follows: w := w - lr * dw; b := b - lr * db. Thus, looping over for n_iter, we hope to reach a convergence point where the cost function won't decrease any further. 

#### How to use this solution
```git clone https://github.com/rezacsedu/Celonis_DS_Task.git
cd data/
unzip UWaveGestureLibraryAll.zip```

Then from within the 'Celonis_DS_Task' directory, run the following from command line:

```python main.py``` 

It will show the results on console. Besides, the convergence of the training losses for the numpy based logistic regression will be generated in the imgs folder with the 'training_loss.png' file name. 

A supporting notebook [Classification_with_manual_FE.ipynb](https://github.com/rezacsedu/Celonis_DS_Task/blob/main/Classification_with_manual_FE.ipynb) is provided, which shows how the baseline logistic regression model works compared to other classifiers (e.g., logistic regression in sklearn and TimeSeriesForestClassifier from the SKTIME library). In particular, my logistic regression model yields accuracy of 54%, underperforming the sklearn-based logistic regression and sktime-based TimeSeriesForestClassifier that yielded accuracy of 69% and 90%, respectively. 

On the other hand, the sample classifier clearly outperforms itself when it ws trained on the features extracted with the sktime library. In particular, my logistic regression model this time yields accuracy of 84%, underperforming the sklearn-based logistic regression that yielded accuracy of 81%. On the other hand and sktime-based TimeSeriesForestClassifier that yielded accuracy of 97%. Please refer to the supporting notebook [Classification_with_SKTIME_FE.ipynb](https://github.com/rezacsedu/Celonis_DS_Task/blob/main/Classification_with_SKTIME_FE.ipynb). This signifies that feature engineering has significant impacts on the performance of individual models. 

#### Task 1.3: Training other models using AutoML
I tried with PyCaret - one of my favourite AutoML library. I found that the extra tree and randome forest classifiers turns to be the best model for this dataset, giving f1 score of over 97%, even outperforming the result reportted in the original paper by 1%! To justiofiy this, I have prepared a supporting notebook  [PyCaret_with_FE_SKTIME.ipynb](https://github.com/rezacsedu/Celonis_DS_Task/blob/main/PyCaret_with_FE_SKTIME.ipynb). 

To give the quick impression, let's see how different classifiers performed on the data: 

<p align="center"><img src="imgs/pycaret_compare_model.png?" width="700" height="350"></p>

Let's see how did extra tree classifier performed across different gesture types w.r.t confusion matrix:

<p align="center"><img src="imgs/cm.png?" width="600" height="300"></p>

As we can see it mostly made correct prediction and was less confused among classes. Further, we plot the ROC curve

<p align="center"><img src="imgs/roc.png?" width="600" height="300"></p>

The above ROC signifies the AUC scores were very consistent across different folds, which is a sign of high generalizability. Now let's see how the decision boundary for the extra tree classiifer based on two features:

<p align="center"><img src="imgs/decision_boundary.png?" width="650" height="350"></p>

As we can see that, except for a few samples, extra tree classifer manage to create a clear decision boundary for different types of gesture. IN such a scenario, a linear SVM (with a linear kernel) is also expected to work better, but actually it didn't. 

To further compare, I created another notebook with PyCaret but the models were trained on the features [PyCaret_with_FE_Numpy.ipynb](https://github.com/rezacsedu/Celonis_DS_Task/blob/main/PyCaret_with_FE_Numpy.ipynb) extracted with pure numpy. Let's see how did extra tree classifier performed across different gesture types w.r.t confusion matrix:

<p align="center"><img src="imgs/cm_2.png?" width="600" height="250"></p>
As we can see the classifer made more wrong prediction than that when features were extracted using sktime libarray, making it more confused among classes. Now let's see how the decision boundary for the extra tree classiifer based on two features:

<p align="center"><img src="imgs/decision_boundary_2.png?" width="650" height="300"></p>

As we can see the decision boundary for the extra tree classifer manage is quite different, perhaps due to different way of stacking of the samples and quality of the features extracted. 

### Task 2: How to make a product ready ML software?
As just training and evaluating a model is of no use unless it can be used as a web application, e.g., own website, cloud, or production ready environment. Therefore, we need to think how a model and its associated workflow/pipeline can be converted as a ML software product. 

  - How would you design a devops pipeline using e.g. Github Actions for a Python package? Which functionalities would you include to ensure code quality and consistency?
  - Assuming the pipeline you implemented will be deployed as a product. Now the customer also wants to enable real time classification and consume an API that returns
the classification results. How would you fit that into the existing architecture?
  - The whole system has been a huge success and also other customers want to use it. How would you adapt everything to be able to serve multiple customers  ==, especially keep in mind scalability and data privacy.
  - What would you recommend to automatically transfer machine learning models to production by running microservices for inferencing?

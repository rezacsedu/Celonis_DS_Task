#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from pathlib import Path
from pathlib import Path
from utils.data_util import *
from lr import LogisticRegression as LR
from sklearn.model_selection import KFold, train_test_split
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Feature extraction from the uWaveGesture dataset 
print("Feature extraction started....")
print("============================================================================")
BASE_PATH = 'data/uWaveGestureLibrary/'
gesture_1 = extract_gesture(BASE_PATH + '/U1/')
gesture_2 = extract_gesture(BASE_PATH + '/U2/')
gesture_3 = extract_gesture(BASE_PATH + '/U3/')
gesture_4 = extract_gesture(BASE_PATH + '/U4/')
gesture_5 = extract_gesture(BASE_PATH + '/U5/')
gesture_6 = extract_gesture(BASE_PATH + '/U6/')
gesture_7 = extract_gesture(BASE_PATH + '/U7/')
gesture_8 = extract_gesture(BASE_PATH + '/U8/')

X, y = create_dataset([gesture_1, gesture_2, gesture_3, gesture_4, gesture_5, gesture_6, gesture_7, gesture_8], shuffle=True)
print("Number of samples: " +  str (X.shape[0]))
print("Number of features: " +  str (X.shape[1]))
print("Number of classes: 8")
print("Done!")

# We use 70% of the data for the training and 30% for testing the trained model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the baseline logistic regression model. 
lr = LR(n_iter = 50000)  

print("\nBuilding logistic regression model .....")
print("============================================================================")
lr.fit(X_train, y_train, batch_size=64, lr=0.001, verbose=True)
print("TRaining finished! Now scoring .....")
lr.score(X_train, y_train)

# Let's check how the model's training loss convergences with iterations. 
print("Now plotting the training loss across iterations .....")
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(lr.loss)), lr.loss)
plt.title("Convergence of training loss")
plt.xlabel("#Iterations")
plt.ylabel("Loss")
plt.show()
fig.savefig('imgs/training_loss.png', dpi=fig.dpi)

# We can see that the loss indeed decreases over the iteration, this means training for more iteration could further help reduce the training loss. Let's check how the model' performs on test set.  
print("\nEvaluating the trained model on test set ....")
print("============================================================================")
y_pred = lr.predict_classes(X_test) 
accuracy_score(y_test, y_pred)

# A decrease of the testing loss of 7% is a sign of potential overfitting - a phenomena for the which the LR model didn't work well on unseen data. Perhaps adding regularization techniques could help. or perhaps shuffling the data ....!
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix on the test set")
print(confusion_matrix)

# We can see the model was pretty much confused at correctly recognising different gesture signals. To further diagnose the reason, let's see the class-wise classsificatin report. 
print("Class-wise classsificatin report on the test set")
print(classification_report(y_test, y_pred))

print("Class-wise classsificatin report on the test set")
print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
print("Done!")

# From the class-wise classsificatin report, it is clear the model is no better than ramdom at correctly recognising different gesture signals except for class 1, 4, 6, and 7. Multinominal logistic regression from the sklearn library. 
print("\nTraining and evaluating the multinominal logistic regression model from sklearn....")
print("============================================================================")
lr_sk = LogisticRegression(multi_class='multinomial', max_iter=5000)
lr_sk.fit(X_train, y_train)
lr_sk.score(X_train, y_train)
y_pred = lr_sk.predict(X_test) 
print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
print("Done!")

# Yielding an f1 score of 54% made the sklearn-based logistic regression classifier is comparable of the baseline implementation of my numpy implementation of logistic regression. 
# Now that none of these two classifiers are no good than random, let's try to do the clasifcation using the time series forecast classifier from the SKTIME library (https://github.com/alan-turing-institute/sktime). This will give us an indication if we should try out with other libraries or models. 
# The sktime is a library for time series analysis in Python. It provides a unified interface for multiple time series learning tasks. Currently, this includes time series classification, regression, clustering, annotation and forecasting. It comes with time series algorithms and scikit-learn compatible tools to build, tune, and validate time series models. 
print("\nTraining and evaluating the TimeSeriesForestClassifier model from sktime....")
print("============================================================================")
tc = TimeSeriesForestClassifier()
tc.fit(X_train, y_train)
y_pred = tc.predict(X_test)
accuracy_score(y_test, y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
print("Done!")

# Summary: the TimeSeriesForestClassifier clearly outperforms both versions of the logistic regression, giving an indication that trying out with other libraries or models could further enhnce the classiifcation accuracy.

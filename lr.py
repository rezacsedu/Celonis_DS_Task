#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created by: Md. Rezaul Karim
# Created date: 16/04/2022
# version ='1.0'

import numpy as np

class LogisticRegression:
    """ A simple baseline implementation of logistic regression, without any regularizers, etc. Besides, I tried to follow scikit-learn 
       -- where variable with trailing underscore is designed to have value after the fit() method is called for example. """ 

    def __init__(self, n_iter = 50000, threshold=1e-3):
        self.n_iter = n_iter
        self.threshold = threshold
    
    def fit(self, X, y, batch_size=128, lr=0.001, rand_seed=4, verbose=False): 
        """We just randomly initialize parameters 'w' and 'b' with 0 to start with (w vector has the shape (945, 1)), whereas the values will be updated using the optimization function."""

        np.random.seed(rand_seed) 
        self.classes = np.unique(y)
        self.class_labels = {c:i for i, c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        self.fit_(X, y, batch_size, lr, verbose)
        return self
 
    def fit_(self, X, y, batch_size, lr, verbose):
        """ We optimize the cost function (cross-entropy) and update the parameters to minimize the cost. Using the gradients descent algorithm, 
	    the gradients of the 'w' and 'b' parameters are computed and the parameters are updated, as follows: w := w - lr * dw; b := b - lr * db. 
        Thus, looping over for n_iter, we hope to reach a convergence point where the cost function won't decrease any further. """
	
        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.threshold: break
            if i % 1000 == 0 and verbose: 
                print(' Training accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i +=1
    
    def predict(self, X):
        return self.predict_(self.add_bias(X))
    
    def predict_(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1, len(self.classes))
        return self.softmax(pre_vals)
    
    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))  
		
    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)] 
   
    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)
    
    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))

    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)
 
    def cross_entropy(self, y, probs):
        """ Cross-entropy between two probability distributions of y and p_pred """ 

        return -1 * np.mean(y * np.log(probs))

    def softmax(self, z):
        """ This is the activation function based on which the predictions will be made. It takes z - product of features with weights 
	and returns discrete class over the classes. In fact, this makes the logistic regression different from linear regression. 
        """

        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

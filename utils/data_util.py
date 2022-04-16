import os
import numpy as np
import sktime
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_2d_np_array
from pathlib import Path

def extract_gesture(directory):
    gesture = []
    #label = []
    pathlist = Path(directory).rglob('*.txt')
    print(pathlist)
    
    for path in pathlist:
        print(path)
        file = open(path, "r")
        lines = file.readlines()[1:]
        file.close()
        x_array = []
        y_array = []
        z_array = []
        for line in lines:
            s = line.split()
            x_array.append(float(s[0]))
            y_array.append(float(s[1]))
            z_array.append(float(s[2]))
            
        gesture.append(np.concatenate(([x_array], [y_array], [z_array]), axis=0))
        #label.append(user_index)
    return gesture

def create_dataset(gestures, shuffle=False):
    n_samples = 0
    n_features = 0
    for gesture in gestures:
        n_samples += len(gesture)
        for sample in gesture:
            if n_features < sample.shape[1]:
                n_features = sample.shape[1]
    X = np.zeros((n_samples, 3 * n_features))
    y = np.zeros(n_samples, dtype=int)
    i = 0
    for index, gesture in enumerate(gestures):
        for sample in gesture:
            X[i, :sample.shape[1]] = sample[0, :]
            X[i, n_features:n_features+sample.shape[1]] = sample[1, :]
            X[i, 2*n_features:2*n_features+sample.shape[1]] = sample[2, :]
            y[i] = int(index)
            i += 1
    if shuffle:
        p = np.random.permutation(n_samples)
        X = X[p, :]
        y = y[p]
    return X, y

def getFeatureSKTime(TRAIN_PATH, TEST_PATH):
    X_train, y_train = load_from_tsfile_to_dataframe(TEST_PATH)
    X_test, y_test = load_from_tsfile_to_dataframe(TRAIN_PATH)
    X_train = from_nested_to_2d_np_array(X_train)
    X_test = from_nested_to_2d_np_array(X_test)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_test, y_test

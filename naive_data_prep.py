#!/usr/bin/env python
# coding: utf-8

import requests
import os
import pyunpack
import pandas as pd
import re
import numpy as np


def download_data(remote_url,data_dir):
  # Function to download the Zip file given and save it to a specified directory
  os.mkdir(data_dir)
  data_file = os.path.join(data_dir, 'uWaveGestureLibrary.zip')
  data = requests.get(remote_url)
  with open(data_file, 'wb')as file:
    file.write(data.content)


def unzip_data(data_dir):
  # Function to extract Zip file and rar files recursively
  arch = pyunpack.Archive(os.path.join(data_dir,'uWaveGestureLibrary.zip'))
  arch.extractall(directory=data_dir)
  os.remove(os.path.join(data_dir,'uWaveGestureLibrary.zip'))

  for root, dirs, files in os.walk(data_dir):
      for filename in files:
          if filename.endswith(".rar") :
              print('Extracting '+os.path.join(root,filename))
          else:
              print('Removing '+os.path.join(root,filename))
              os.remove(os.path.join(root,filename))
          if filename.endswith(".rar"):
              name = os.path.splitext(os.path.basename(filename))[0]
              try:
                  arch = pyunpack.Archive(os.path.join(root,filename))
                  item_dir = os.path.join(root,name)
                  os.mkdir(item_dir)
                  arch.extractall(directory=item_dir)
                  os.remove(os.path.join(root,filename))
              except Exception as e:
                  print("ERROR: BAD ARCHIVE "+os.path.join(root,filename))
                  print(e)              


def extract_data(data_dir):
  # Function to extract data from text files and load them to a pandas DataFrame
  for root, dirs, files in os.walk(data_dir):
      data_df = pd.DataFrame(columns=['x-acc','y-acc','z-acc','gesture','repetition','item_id'])
      item_id = 1
      for filename in files:
        if filename.endswith('.txt') and 'Template_Acceleration' in filename:
              item_df = pd.read_table(os.path.join(root, filename),delimiter = ' ',header=None, names=['x-acc','y-acc','z-acc','gesture','repetition'])
              size = len(item_df)
              m = re.search('Template_Acceleration(.+?).txt', filename)
              ges_rep = m.group(1).split('-')
              gesture = int(ges_rep[0])
              if ges_rep[1] != '':
                repetition = int(ges_rep[1])
              else:
                repetition = np.nan
              item_df['gesture'] = [gesture] * size
              item_df['repetition'] = [repetition] * size
              item_df['item_id'] = [item_id] * size
              data_df = data_df.append(item_df)

              item_id +=1
  return data_df


def get_data(remote_url,data_dir):
  # Function to complete data preparation 
  download_data(remote_url,data_dir)
  unzip_data(data_dir)
  data_df = extract_data(data_dir)
  X = np.array(data_df[['x-acc','y-acc','z-acc']])
  y = np.array(data_df['gesture'])
  return X,y


remote_url = 'http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip'
data_dir = './celonis'
X,y = get_data(remote_url, data_dir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras.utils import np_utils

print('Dota Loading ...')

# Load of buildings energy data from file
dataset = pd.read_csv('/Users/theodosismoumiadis/Documents/ENB2012_data.csv')
# A command that returns all data rows in random order
# dataset = dataset.sample(frac=1)

# Separation of training data into X and Y sets
X_train = dataset.ix[0:667,1:9].values.astype('float32')
Y1_train = dataset.loc[0:667,'Y1'].values.astype('float32')
Y2_train = dataset.loc[0:667,'Y2'].values.astype('float32')

# Separation of test data into X and Y sets
X_test = dataset.ix[668:767,1:9].values.astype('float32')
Y1_test = dataset.loc[668:767,'Y1'].values.astype('float32')
Y2_test = dataset.loc[668:767,'Y2'].values.astype('float32')

print('Data Loaded')

print(' ')
print('Data Normalization ... ')

# Training and test input data Normalization
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std

X_test -= mean
X_test /= std

print('Data Normalized')

print(' ')
print('Neural network training ... ')

# NN model definition

# Train of the DL network using tf.keras
model = tf.keras.models.Sequential()
# The Input Layer:
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(8,)))
# The Hidden Layers
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))

# The Output Layer
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit(X_train, Y1_train, epochs=300, batch_size=10, verbose=0)

test_mse_score, test_mae_score = model.evaluate(X_test, Y1_test)


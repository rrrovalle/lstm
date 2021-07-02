#NOT DONE YET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

#'https://raw.githubusercontent.com/cristmelo/PracticalGuide/master/1.%20Data%20Set/all_releases_no_repetitions.csv'
url = 'raw'
dataset_train = pd.read_csv(url)

#select features
cols = list(dataset_train)[2:12]
 
#extact change_probability
change_prob_train = list(dataset_train['change_probability'])
 
#print
print(f'Training set shape  \n{dataset_train.shape}')
print(f'All chagne probs  \n{dataset_train}')
print(f'All chagne probs  \n{cols}') 
 
# removing all commans and convert data to matrix shape format
dataset_train = dataset_train[cols].astype(str)
for i in cols:
  for j in range(0, len(dataset_train)):
    dataset_train[i][j] = dataset_train[i][j].replace(',','')
dataset_train = dataset_train.astype(float)

#using mult predict (feat) 
train_set = dataset_train.to_numpy()

print(f'Shape of training set \n{train_set.shape}')
#train_set
  
#feature scaling
sc = StandardScaler()
train_set_scale = sc.fit_transform(train_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(train_set[:,0:1])

#set x and y
X_train = []
y_train = []

n_future = 60  #number of predicts we want into the future
n_past = 90    #number of past data we want to use to predict future

for i in range(n_past, len(train_set_scale)- n_future +1):
  X_train.append(train_set_scale[i-n_past:1, 0:dataset_train.shape[1]-1])
  y_train.append(train_set_scale[i+n_future:i + n_future,0])

X_train, y_train = np.array(X_train), np.array(y_train)

print(f'Shape of training set \n{X_train.shape}')
print(f'Shape of training set \n{y_train.shape}')

## TO-DO ##
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))
# model.add(LSTM(10,return_sequences=False))
# model.add(Dropout(0.25))
# model.add(Dense(1,activation='linear'))
# model.compile(optimizer='adam',loss='mse')

# #summary
# model.summary()

#fit
#model.fit(X_train, y_train,epochs=200,verbose=1)

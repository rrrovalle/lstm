import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from numpy import array

#https://raw.githubusercontent.com/cristmelo/PracticalGuide/master/1.%20Data%20Set/all_releases_no_repetitions.csv

url = 'raw'
df1 = pd.read_csv(url)

#split samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
    
#split data
raw_seq = df1['LOC']
n_steps = 3
X, y    = split_sequence(raw_seq, n_steps)

#features
n_features = 1
X = X.reshape(X.shape[0], X.shape[1],n_features)

#vanilla model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(n_steps,n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

#fit
model.fit(X,y,epochs=200,verbose=0)

#example 
x_input = array([7,1,9])
x_input = x_input.reshape((1,n_steps, n_features))
yhat = model.predict(x_input,verbose=0)
#print predict
print(yhat)

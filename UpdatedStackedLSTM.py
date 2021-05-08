### Data Collection
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import numpy as np
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from numpy import array

import math
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)



def LSTMPrediction(tick):
    import matplotlib.pyplot as plt
    df=pd.read_csv(f'{tick}.csv')
    data_training = df.drop(['date'], axis=1)
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(data_training)

    training_size = int(len(data_training) * 0.65)
    test_size = len(data_training) - training_size
    train_data, test_data = data_training[0:training_size, :], data_training[training_size:len(data_training), :5]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100

    X_train = []
    y_train = []

    for i in range(time_step, train_data.shape[0]):
        X_train.append(train_data[i - time_step:i])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []

    for i in range(time_step, test_data.shape[0]):
        X_test.append(test_data[i - time_step:i])
        y_test.append(test_data[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)


    #X_train, y_train = create_dataset(train_data, time_step)
    #X_test, ytest = create_dataset(test_data, time_step)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 5)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=5, batch_size=64, verbose=1)



    y_pred = model.predict(X_test)

    scale = 1 / 8.18605127e-04
    y_pred = y_pred * scale
    y_test = y_test * scale


    #plt.figure(figsize=(14, 5))
    #plt.plot(y_test, color='red', label='Real Google Stock Price')
    #plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
    #plt.title('Google Stock Price Prediction')
    #plt.xlabel('Time')
    #plt.ylabel('Google Stock Price')
    #plt.legend()
    plt.show()


    print(len(test_data))
    #x_input = test_data[(len(test_data) - 100):].reshape(1, -1)
    x_input = test_data[(len(test_data) - 100):]

    print(x_input)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while (i < 30):

        if (len(temp_input) > 100):
            ## print(temp_input)
            x_input = np.array(temp_input[1:])
            # print("{} day input {}".format(i, x_input))
            #->x_input = x_input.reshape(1, -1)
            #->x_input = x_input.reshape((1, n_steps, 5))
            ## print(x_input)
            yhat = model.predict(x_input)
            # print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            ## print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            #->x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1

    # print(lst_output)

    day_new = np.arange(2, 101)
    day_pred = np.arange(101, 131)

    plt.plot(day_new, (df1[(len(df1) - 99):])*scale)
    plt.plot(day_pred, (lst_output)*scale)





def DoMultiplestocks():
    tickers=['TRH']
    #count=0
    for tick in tickers:
        LSTMPrediction(tick)
        #print(tick)
        #count=count+1


DoMultiplestocks()
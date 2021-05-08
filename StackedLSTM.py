### Data Collection
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import numpy as np
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Bidirectional,Dropout
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
    #print(df.head())
    df1=df.reset_index()['adjclose']
    #print(df1)
    #plt.plot(df1)
    #plt.show()
    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    #print(df1)
    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    #print(X_train.shape), print(y_train.shape)
    #print(X_test.shape), print(ytest.shape)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model=Sequential()
    model.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(100,1))))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.4))
    model.add(Dense(1,activation="linear"))

    #for non Biderctional and non dropout
    '''
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    '''

    model.compile(loss='mean_squared_error',optimizer='adam')

    #print(model.summary())
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    ### Calculate RMSE performance metrics
    print("mean square error of train set")
    print(math.sqrt(mean_squared_error(y_train, train_predict)))

    ### Test Data RMSE
    print("mean square error of test set")
    print(math.sqrt(mean_squared_error(ytest, test_predict)))

    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)



    ### Plotting
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot =  np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    #plt.plot(scaler.inverse_transform(df1))
    #plt.plot(trainPredictPlot)
    #plt.plot(testPredictPlot)
    #plt.show()


    #len(test_data)
    x_input=test_data[(len(test_data)-100):].reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while (i < 30):

        if (len(temp_input) > 100):
            ## print(temp_input)
            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            ## print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            ## print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1

    #print(lst_output)

    day_new=np.arange(2,len(df1))
    day_pred=np.arange(len(df1),len(df1)+30)


    plt.plot(day_new,scaler.inverse_transform(df1[:(len(df1)-2)]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))

    plt.savefig(f'{tick}.png', bbox_inches='tight')
    plt.close()

    #plt.show()


def DoMultiplestocks():
    tickers=['NLIC','SCB','NICA','NLICL','SBI','tsla','UPPER']
    #count=0
    for tick in tickers:
        LSTMPrediction(tick)
        #print(tick)
        #count=count+1


DoMultiplestocks()
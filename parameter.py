import os
import time
from tensorflow.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 4 #changed from 15 to 5

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
#SHUFFLE = False
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
#SPLIT_BY_DATE = False
SPLIT_BY_DATE = True #changed from prev False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high","low", "stoch", "RSI","rsidata","macd"]
#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
#date_now = time.strftime("%Y-%m-%d")
date_now = '2021-01-25'
#print(date_now)

### model parameters

N_LAYERS = 3#changd from original 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 20% dropout
DROPOUT = 0.2
# whether to use bidirectional RNNs
BIDIRECTIONAL = True

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 50
EPOCHS = 50 #Changed from prev 500 to 10

# Amazon stock market
#ticker = "AMZN"
ticker = "NABIL"
activationfunction=["linear","tanh","sigmoid","relu","elu","softmax","softplus","softsign","exponential",""]



ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
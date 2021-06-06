import os
import time
from tensorflow.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 26 #128 imp number
# Lookup step, 1 is the next day
LOOKUP_STEP = 3 #changed from 15 to 5

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
#SHUFFLE = False
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
#SPLIT_BY_DATE = False
SPLIT_BY_DATE = False #changed from prev False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use

#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high","low", "volatility_atr", "volatility_bbm","volatility_bbh","volatility_bbl","volatility_bbw","volatility_kch","volatility_kcl","volatility_kcw","WilliamR","macd","macddiff","momentum_stoch_rsi","momentum_stoch_rsi_d","momentum_stoch_rsi_k","stoch","RSI"]
#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]#original
#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high","low", "volatility_atr", "volatility_bbm","volatility_bbh","volatility_bbl","volatility_bbw","volatility_kch","volatility_kcl"]  #Accordingtovolatility
#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high","low", "macd","macddiff","momentum_stoch_rsi","momentum_stoch_rsi_d","momentum_stoch_rsi_k","stoch","RSI"]#Accordingtomomentum
#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high","low", "macd", "macddiff","RSI"]#AccordingtoTrend
#FEATURE_COLUMNS = ["adjclose", "volume", "open", "high","low", "volatility_atr", "macddiff","RSI", "WilliamR","volatility_bbw"]#Mixed

#best upto now
#FEATURE_COLUMNS = ["adjclose", "volume", "high","low", "WilliamR" ,"macddiff","momentum_stoch_rsi","stoch","RSI","volatility_atr","volatility_bbw"]#Accordingtomomentumwithwilliam  #BEST upto now
#FEATURE_COLUMNS = ["adjclose", "volume", "IchiBaseLine","IchiConversionLine","IchiLineA","IchiLineB","EMA200","macddiff"]
#FEATURE_COLUMNS = ["adjclose", "volume", "WilliamR" ,"macddiff","momentum_stoch_rsi","stoch","RSI","volatility_atr","volatility_bbw"]#new
FEATURE_COLUMNS = ["adjclose"]

# date now
date_now = time.strftime("%Y-%m-%d")
#date_now = '2021-01-25'
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
BATCH_SIZE = 40 #32 imp number
EPOCHS = 6 #Changed from prev 500 to 10

# Amazon stock market
#ticker = "AMZN"
ticker = "NEPSE"
activationfunction=["elu","linear","tanh","sigmoid","relu","softmax","softplus","softsign","exponential","sin"]



ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
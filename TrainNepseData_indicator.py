from train_indicator import *
from test_indicator import *

#tickersloop=["NTC","NRIC","BOKL","EBL","GBIME","LIFEINSU","UPPER","NICA","NLIC","NMB"]
#tickersloop=["NEPSE","BANKING","NLIC","LICN","SBL","GBIME","LIFEINSU","NMB","SHIVM","NICA","NABIL"]
#tickersloop=["NICA","NMB","NEPSE","BANKING","GBIME","UPPER","ADBL","SHIVM"]
#tickersloop=["NICA","NMB","NEPSE_index","Banking_index","UPPER","NABIL","SHIVM"] #My stocks
tickersloop=["PRVU","SHIVM","NEPSE_index","UPPER","DDBL","CHCL"]
#trainingfunction(tickersloop[0])

for tickers in tickersloop:
    trainingfunction(tickers) #for individual test
    Test_function(tickers)
    '''try:
        trainingfunction(tickers)
        Test_function(tickers)
    except:
        print("Error in: ",tickers)'''

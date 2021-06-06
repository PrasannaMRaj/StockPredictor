import requests
import pandas as pd
import csv
import json
import os


def start_requests():

    #url = "https://www.neabilling.com/viewonline/viewonlineresult"

    url = "https://www.neabilling.com/viewonline/viewonlineresult/?NEA_location=207&sc_no=052.23.012&consumer_id=9812&Fromdatepicker=01/01/2021&Todatepicker=3/9/2021"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'}



    Fromdatepicker = "04/01/2021"
    Todatepicker = "5/12/2021"
    SCNo="052.23.012"
    NEALocation="207"


    payload = {"NEA_location": NEALocation, "sc_no":SCNo , "consumer_id" : 9812,"Fromdatepicker": Fromdatepicker, "Todatepicker": Todatepicker}

    page = requests.get(url, headers=headers)
    print (page)
    #print(page.json())
    df = pd.DataFrame(page.json())
    print(df)


start_requests()
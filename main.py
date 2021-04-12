import numpy as np
import datetime
import pandas as pd
import os

from regressor import run


def loadDataFromDataset():
    #citirea datelor
    df = pd.read_csv('./weatherHistory/weatherHistory.csv')

    #eliminarea coloanei Loud Cover deoarece nu este necesara cerintelor noastre de implementare
    del df["Loud Cover"]

    #extragerea lunii din acest format de data
    df["Date"] = pd.to_datetime(df["Formatted Date"], utc=True) #Return UTC DatetimeIndex,Convert argument to datetime.

    #crearea unei coloane noi pentru luna
    df["Month"] = df["Date"].dt.month

    #stergerea coloanelor de data cu acest format
    del df["Formatted Date"]
    del df["Date"]

    return df

df=loadDataFromDataset()
run(df)


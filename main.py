import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

Data = pd.read_csv('boop.csv', encoding='cp1252')
Data.info()
Data.describe().transpose()

print("Mean = ", Data.mean(axis=0))
print("Std deviation = ", Data.std(axis=0))

data_scaled = preprocessing.scale(Data)
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(Data)
print("\nMin max scaled data:\n", data_scaled_minmax)
##############################################################

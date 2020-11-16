import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

input_data = pd.read_csv('boop.csv', encoding='cp1252', delimiter=None)

input_data.info()
input_data.describe().transpose()

print(input_data.head(5))
label_year = input_data['rok']
label_sprzedaz_detaliczna_ceny = input_data['sprzedaz_detaliczna_ceny']
label_sprzedaz_detaliczna_mln_zl = input_data['sprzedaz_detaliczna_mln_zl']
label_powierzchnia_sprzedazowa_sklepow_tys_m = input_data['powierzchnia_sprzedazowa_sklepow_tys_m']
label_liczba_sklepow = input_data['liczba_sklepow']
label_przychody_dzialalnosci_gastronomicznej_mln_zl = input_data['przychody_dzialalnosci_gastronomicznej_mln_zl']
print(label_year, label_sprzedaz_detaliczna_ceny)

"""print("Mean = ", input_data.mean(axis=0))
print("Std deviation = ", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

print('Size of a data frame is :', data.shape)
print(input_data[0:5])

print(input_data.count().sort_values())

data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1')
print("\nL1 normalized data:\n", data_normalized_l1)

# Normalize data
data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2')
print("\nL2 normalized data:\n", data_normalized_l2)"""

train, test = train_test_split(input_data, test_size=0.2, random_state=42, shuffle=True)
print(train.shape, test.shape)
print(train.nunique())


import catboost as cat
cat_feat = ['przychody_dzialalnosci_gastronomicznej_mln_zl', 'liczba_sklepow', 'powierzchnia_sprzedazowa_sklepow_tys_m',
            'rok']
features = list(set(train.columns)-set(['sprzedaz_detaliczna_mln_zl']))
target = 'sprzedaz_detaliczna_mln_zl'
model = cat.CatBoostRegressor(random_state=100, cat_features=cat_feat, verbose=0)
model.fit(train[features], train[target])

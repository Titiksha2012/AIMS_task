# <--------------------------------------SCRIPT FOR 2 ENCODERS--------------------------------------->

import pandas as pd
import numpy as np

def encoder1(df):
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns)
    return df_encoded

def encoder2(df):
    # Label Encoding
    for column in df.select_dtypes(include='object').columns:
        df[column], _ = pd.factorize(df[column])
    return df

data = pd.read_csv('dataset.csv')

data_encoded1 = encoder1(data.copy())

data_encoded2 = encoder2(data.copy())

print("Encoded Dataset 1:")
print(data_encoded1.head())

print("\nEncoded Dataset 2:")
print(data_encoded2.head())



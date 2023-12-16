# <-------------------------------------SCRIPT FOR 2 IMPUTERS---------------------------------------->

import pandas as pd
import pandas as np

def imputer1(df):
    # Replace missing values with the mean
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))

    # For categorical columns, replace missing values with the most frequent value
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

    return df

def imputer2(df):
    # Replace the missing values with the median
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

    # For categorical columns, replace missing values with a constant value 
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna('Unknown'))

    return df

data = pd.read_csv('dataset.csv')

data_imputed1 = imputer1(data.copy())

data_imputed2 = imputer2(data.copy())

print("Imputed Dataset 1:")
print(data_imputed1.head())

print("\nImputed Dataset 2:")
print(data_imputed2.head())
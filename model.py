import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


# Cargar el dataset
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ["Age", "Sex", "Embarked", "Survived"] # Solo incluir 4 features
df_ = df[include]


# 'Sex' y 'Embarked' son variables categoricas con valores no numericos.
# 'Age' tiene NaNs y estos tienen que ser procesados.
# Normalmente no deberian llenar NaNs con un mismo valor (0).
categoricals = []

for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)


# Estamos haciendo one hot encodings de las variables no numericas
# Esto va a crear una nueva columna por cada combinacion de columna/valor
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)


# NUNCA HAGAN ESTO
# Siempre usen un train/val/test split cuando tengan suficiente data
# Cuando tengan poca data usen k-fold cross validation 
# o augmenten la data (mas complicado)
y_hat = 'Survived'
x = df_ohe[df_ohe.columns.difference([y_hat])]
y = df_ohe[y_hat]
lr = LogisticRegression()
lr.fit(x, y) 



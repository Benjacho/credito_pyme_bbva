from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def preprocess_transformers(y_train, transf):
    if transf != 'ln':
        if transf == 'minmax':
            scaler = MinMaxScaler()
            scaler2 = MinMaxScaler()
        elif transf == 'standard':
            scaler = StandardScaler()
            scaler2 = StandardScaler()
        elif transf == 'robust':
            scaler = RobustScaler()
            scaler2 = RobustScaler()
        elif transf == 'boxcox':
            scaler = PowerTransformer(method='yeo-johnson')
            scaler2 = PowerTransformer(method='yeo-johnson')

        mm_scaler2 = scaler2.fit(y_train)
        y_train = mm_scaler2.transform(y_train)
    else:
        # y_train = y_train.values
        y_train = np.log(y_train).values
        mm_scaler = ''
        mm_scaler2 = ''

    return y_train, mm_scaler, mm_scaler2


def transformacion_inversa(y_predict, mm_scaler2):
    if mm_scaler2 != '':
        y_predict = mm_scaler2.inverse_transform(pd.DataFrame(y_predict))
    else:
        y_predict = np.exp(y_predict)
        # y_predict = y_predict

    return y_predict


def predict_model(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    prepared_df, scaler = preprocess_transformers(df, 'minmax')
    y_pred = model.predict(prepared_df)

    transformacion_inversa(y_pred, scaler)
    return y_pred

import pandas as pd
import numpy as np


def preprocess_transformers(data):
    return 0


def predict_model(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    # Preprocess the model
    prepared_df = preprocess_transformers(df)
    y_pred = model.predict(prepared_df)
    return y_pred

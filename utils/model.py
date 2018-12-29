import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss


def custom_metric(date, pred_proba, num_target, universe):
    y = pred_proba*2 - 1
    r = num_target.clip(-1,1) # get rid of outliers
    x = y * r * universe
    result = pd.DataFrame({'day' : date, 'x' : x})
    x_t = result.groupby('day').sum().values
    return np.mean(x_t) / np.std(x_t)


def evaluate_model(df, target, train_index, test_index, params):
    # model = XGBClassifier(**params)
    model = LGBMClassifier(**params)
    model.fit(df.iloc[train_index], target.iloc[train_index])
    return log_loss(target.iloc[test_index], model.predict_proba(df.iloc[test_index]))

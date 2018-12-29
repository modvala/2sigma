import pandas as pd
import numpy as np
from multiprocessing import Pool
import gc

CAT_COLS = ['assetCode']
NUM_COLS = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
            'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
            'returnsOpenPrevMktres10']


def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline', 'firstCreated', 'sourceTimestamp',
    ]
    news_train.drop(drop_list, axis=1, inplace=True)

    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques

    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train


def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df


def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack


def group_news(news_frame):
    news_frame['date'] = news_frame.time.dt.date  # Add date column

    aggregations = ['mean']
    gp = news_frame.groupby(['assetCode', 'date']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date']}
    return gp.astype(float_cols)


def process_date(df):
    df['date'] = df['time'].dt.date
    # df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    # df['day'] = df['time'].dt.day
    df['dayofweek'] = df['time'].dt.dayofweek

    return df


def process_ma(df, columns=['open', 'close', 'volume'], windows=[5, 20, 60]):
    ma_columns = []
    ma_dev_columns = []
    for col in columns:
        for window in windows:
            ma_column = 'ma_{0}_{1}'.format(col, window)
            ma_dev_column = 'ma_dev_{0}_{1}'.format(col, window)
            std_column = 'std_{0}_{1}'.format(col, window)
            ma_columns.append(ma_column)
            ma_dev_columns.append(ma_dev_column)
            df[ma_column] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).mean())
            df[ma_dev_column] = df[col] / df[ma_column] - 1
            df[std_column] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).std())

    return df


def simple_features(data):
    data['price_diff'] = data['close'] - data['open']
    data['close_to_open'] = np.abs(data['close'] / data['open'])
    # news_train_data['sentence_word_count'] = news_train_data['wordCount'] / news_train_data['sentenceCount']
    data['time'] = data['time'].dt.date
    return data


def create_lag(data, features, n_lag=[3, 7, 14, ], shift_size=1):
    # code = data['assetCode'].unique()

    for col in features:
        for window in n_lag:
            rolled = data[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            lag_max = rolled.max()
            lag_min = rolled.min()
            lag_std = rolled.std()
            data['%s_lag_%s_mean' % (col, window)] = lag_mean
            data['%s_lag_%s_max' % (col, window)] = lag_max
            data['%s_lag_%s_min' % (col, window)] = lag_min
            data['%s_lag_%s_std'%(col,window)] = lag_std
            # config.num_cols.extend(['%s_lag_%s_mean', '%s_lag_%s_max', '%s_lag_%s_min', '%s_lag_%s_std'])
    return data.fillna(-1)


def generate_lag_features(data, features, n_lag=[3, 7, 14]):

    assetCodes = data['assetCode'].unique()
    print(assetCodes)
    all_data = []
    datas = data.groupby('assetCode')
    datas = [data[1][['time', 'assetCode'] + features] for data in datas]
    print('total %s data' % len(datas))

    pool = Pool(4)
    all_data = pool.map(create_lag, datas, features, n_lag)

    new_data = pd.concat(all_data)
    new_data.drop(features, axis=1, inplace=True)
    pool.close()

    return new_data


def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data



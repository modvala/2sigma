
# import gc
# import sys

# sys.path.append('./utils/*.*')
from utils.preprocess import *


def prenews(news_train, TEST):
    news_train = preprocess_news(news_train)
    index_df = unstack_asset_codes(news_train)
    if TEST:
        print(index_df.head())
    news_unstack = merge_news_on_index(news_train, index_df)
    del news_train, index_df
    gc.collect()
    if TEST:
        print(news_unstack.head(3))

    news_agg = group_news(news_unstack)
    del news_unstack; gc.collect()
    if TEST:
        print(news_agg.head(3))

    return news_agg


def predata(news_agg, market_train, TEST):
    market_train = process_date(market_train)
    market_train = process_ma(market_train)

    df = market_train.merge(news_agg, how='left', on=['assetCode', 'date'])
    del market_train, news_agg
    gc.collect()
    if TEST:
        print(df.head(3))
    return df
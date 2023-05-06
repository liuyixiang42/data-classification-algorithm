import numpy as np
import pandas as pd
import scipy.signal as sig
from sklearn.preprocessing import MinMaxScaler


def pretreat():
    # 加载数据文件,将日期解析为时间序列格式
    data = pd.read_csv('data.csv', index_col=0, parse_dates=True)

    # 对缺失值进行填充(前向填充)
    data.fillna(method='ffill', inplace=True)

    # 对数据进行差分,进行平稳化处理
    data_diff = data.diff().dropna()

    # 对数据进行归一化处理
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data_diff)

    # 对数据进行平滑处理
    data = data.rolling(window=7).mean()

    return data



def pretreat_to_database():
    data = pd.read_csv('data.csv')

    # 删除缺失值
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # 对数据进行标准化
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return df_scaled


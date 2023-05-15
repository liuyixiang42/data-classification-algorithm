import numpy as np
import pandas as pd
import statsmodels.api as sm

# 读取油气时序数据
data = pd.read_csv('oil_transformer_data.csv', index_col=0, parse_dates=True)

# 数据预处理：对数变换
data = np.log(data)

# 数据拆分：将数据集按照一定比例划分为训练集和测试集
train_data = data.iloc[:80, :]
test_data = data.iloc[80:, :]

# 提取时间序列特征
decomposition = sm.tsa.seasonal_decompose(train_data, period=12)
trend = decomposition.trend
seasonality = decomposition.seasonal
residual = decomposition.resid

# 引入油温作为外生变量
exog_train = pd.concat([trend, seasonality, residual, train_data['oil_temp']], axis=1)

# 模型选择：根据数据的特点，选择适合的VARMA模型
model = sm.tsa.VARMAX(train_data, order=(1, 1), trend='c', exog=exog_train)

# 模型拟合
results = model.fit()

# 引入油温作为外生变量
exog_test = pd.concat([test_data['trend'], test_data['seasonality'], test_data['residual'], test_data['oil_temp']], axis=1)

# 模型预测
pred = results.predict(start=80, end=len(data)-1, exog=exog_test)

# 将预测结果还原成原始数据的尺度
pred_values = np.exp(pred)

# 查看预测结果
print(pred_values)
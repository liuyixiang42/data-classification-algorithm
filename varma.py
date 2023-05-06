import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# 生成随机多变量时间序列数据
np.random.seed(1234)
data = np.random.randn(100, 9)

# 将数据分为训练集和测试集
train_data = data[:80, :]
test_data = data[80:, :]

# 建立 VARMA 模型
model = sm.tsa.VARMAX(train_data, order=(2, 2))

# 拟合模型并进行预测
results = model.fit()
pred = results.predict(start=80, end=99)

# 计算预测误差
mse = np.mean((test_data - pred)**2)

# 打印预测结果和预测误差
print("预测结果：", pred)
print("预测误差：", mse)

show_data = np.concatenate((train_data, pred), axis=0)
for i in range(9):
    plt.plot(show_data[:, i], label=i)

plt.legend()

plt.show()


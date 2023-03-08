import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_csv('data.csv')

# 将数据分为训练集和测试集
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# 对数据进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
test = scaler.transform(test)


# 创建训练和测试数据集
def create_dataset(dataset, look_back=20):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i:(i + look_back), :]
        y = dataset[i + look_back, 0]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


look_back = 20
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 构建LSTM模型
# 首先，创建一个Sequential对象，它是Keras中的一种顺序模型，可以按顺序添加一系列的神经网络层
model = Sequential()
# 添加第一层LSTM,包含100个LSTM单元
# 表示输入的数据是一个三维张量，
# 其中第一个维度是时间步数，第二个维度是输入特征数，第三个维度是样本数。
# 因为是处理多参数时序数据，所以第二个维度是train数据集中的特征数
model.add(LSTM(100, input_shape=(look_back, train.shape[1]), return_sequences=True))
# 添加一个Dropout层，其作用是在训练过程中随机地“丢弃”一些神经元，以防止过拟合。
model.add(Dropout(0.2))
# 添加第二层LSTM，其参数与第一层LSTM类似，但是这里的return_sequences参数设置为True，表示需要保留完整的序列输出。
model.add(LSTM(50, return_sequences=True))
# 再次添加Dropout层。
model.add(Dropout(0.2))
# 添加第三层LSTM，与第二层LSTM参数相同，但这里的return_sequences参数设置为False，表示我们只需要输出最后一个时间步的输出。
model.add(LSTM(50, return_sequences=False))
# 再次添加Dropout层
model.add(Dropout(0.2))
# 最后，添加一个全连接层(Dense)，其中包括一个神经元和线性激活函数，用于输出预测值。
model.add(Dense(1, activation='linear'))

# 编译模型
# 使用均方误差（Mean Squared Error，MSE）作为损失函数,将预测值和真实值之差的平方作为损失，并求平均值。均方误差越小，说明模型的预测越接近真实值
# 易于求解梯度，均方误差的求导比较简单，计算梯度的成本较低，这使得神经网络的训练过程更加高效。
# 对离群值不敏感，均方误差是将预测值和真实值之差的平方作为损失，并求平均值，因此相对于其他损失函数如平均绝对误差（MAE），均方误差对于离群值的影响更小。这使得均方误差在处理带有噪声数据时更加有效。
# 独立于输入维度，均方误差与输入维度无关，因此可以处理各种不同的数据类型和数据形状。这使得均方误差可以应用于各种不同的回归问题中。
# 如果需要更好的解释性能的话可以选择更换损失函数
model.compile(loss='mean_squared_error', optimizer='adam')

# 添加early stopping回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 训练模型
history = model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=2,
                    validation_split=0.1, callbacks=[early_stopping])

# 对测试数据集进行预测
testPredict = model.predict(testX)

# 将预测结果反向标准化
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 计算均方根误差
testScore = np.sqrt(np.mean((testPredict - testY) ** 2))
print('Test Score: %.2f RMSE' % (testScore))

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from keras.optimizers import RMSprop


def decode_one_hot(arr):
    result = []
    for i in range(len(arr)):
        result.append(np.argmax(arr[i]))
    return result


def encode_one_hot(arr, num):
    return to_categorical(arr, num_classes=num)


# 读取原始数据集
directory = 'data'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
csv_file_paths = [directory + '/' + file for file in csv_files]
dfs = [pd.read_csv(file,
                   usecols=['A相', 'B相', 'C相', 'A相.1', 'B相.1', 'C相.1', 'A相.2', 'B相.2', 'C相.2', 'A相.3', 'B相.3',
                            'C相.3', 'A相.4', 'B相.4', 'C相.4']) for file in csv_file_paths]

# 转换为ndarray对象
X = np.array([df.to_numpy() for df in dfs])
# 样本数量，样本长度，样本维度数量和分类数量
num_samples = X.shape[0]
seq_len = X.shape[1]
num_dims = X.shape[2]
num_classes = 5

# 对每个维度进行标准化处理
# 将各个维度的数据值转换为具有相同规模和分布的数据，以避免在模型训练时某些维度对结果的影响过大或过小，进而导致模型过拟合或欠拟合。
for i in range(num_dims):
    scaler = StandardScaler()
    X[:, :, i] = scaler.fit_transform(X[:, :, i])

# 生成随机的分类标签
y_init = np.random.randint(0, num_classes, num_samples)

# 将类别标签转换为one-hot编码，作用是将离散的类别标签转换为连续的数值向量，以便于在神经网络中进行计算和优化
# 使用one-hot编码可以将类别之间的距离明确表示出来，同时可以避免将类别之间的大小关系纳入考虑，从而更加准确地反映类别之间的差异
y = encode_one_hot(y_init, num_classes)

# 定义窗口大小和步长
window_size = 10
step_size = 5

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将训练集和测试集的每个时序数据划分为多个窗口
X_windows_train = []
y_windows_train = []
for j in range(X_train.shape[0]):
    X_window = []
    y_window = []
    for i in range(0, X_train.shape[1] - window_size + 1, step_size):
        X_window.append(X_train[j, i:i + window_size, :])
        y_window.append(y_train[j])
    X_windows_train.append(np.array(X_window))
    y_windows_train.append(np.array(y_window))

X_windows_test = []
y_windows_test = []
for j in range(X_test.shape[0]):
    X_window = []
    y_window = []
    for i in range(0, X_test.shape[1] - window_size + 1, step_size):
        X_window.append(X_test[j, i:i + window_size, :])
        y_window.append(y_test[j])
    X_windows_test.append(np.array(X_window))
    y_windows_test.append(np.array(y_window))

# 建立 LSTM 模型
model = Sequential()
model.add(LSTM(units=64, input_shape=(window_size, num_dims), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(0.01)))
# 自适应学习率，更新模型权重以优化模型
optimizer = RMSprop(lr=0.001, rho=0.9)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
n_windows = X_windows_train[0].shape[0]
for i in range(n_windows):
    print(f'Training window {i + 1}/{n_windows}')
    model.fit(X_windows_train[i], y_windows_train[i], epochs=100, batch_size=32, verbose=0)

# 预测测试集的结果
y_pred_one_hot = []
for i in range(len(X_windows_test)):
    y_pred_window = model.predict(X_windows_test[i])
    pred = []
    for j in range(y_pred_window.shape[1]):
        col_sum = 0
        for k in range(y_pred_window.shape[0]):
            col_sum += y_pred_window[k][j]
        pred.append(col_sum / y_pred_window.shape[0])
    y_pred_one_hot.append(np.array(pred))
y_pred = decode_one_hot(y_pred_one_hot)

# 输出模型评估指标
count = sum(x == y for x, y in zip(y_pred, decode_one_hot(y_test)))
print(f'Accuracy: {count/len(y_pred)}')


import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# 读取原始数据集
directory = 'data'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
csv_file_paths = [directory + '/' + file for file in csv_files]
dfs = [pd.read_csv(file, usecols=['A相', 'B相', 'C相', 'A相.1', 'B相.1', 'C相.1', 'A相.2', 'B相.2', 'C相.2', 'A相.3', 'B相.3', 'C相.3', 'A相.4', 'B相.4', 'C相.4']) for file in csv_file_paths]

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
y = np.random.randint(0, num_classes, num_samples)

# 将类别标签转换为one-hot编码，作用是将离散的类别标签转换为连续的数值向量，以便于在神经网络中进行计算和优化
# 使用one-hot编码可以将类别之间的距离明确表示出来，同时可以避免将类别之间的大小关系纳入考虑，从而更加准确地反映类别之间的差异
y = to_categorical(y, num_classes=num_classes)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM模型，包括三个LSTM层、两个Dense层和四个Dropout层。每个LSTM层都有64个神经元，每个Dense层都有32个或num_classes个神经元
# Dense层作用是将前一层的输出进行线性变换，并加上一个偏置项，然后将结果传递给激活函数进行非线性转换
# Dropout层用于减少过拟合的风险
model = Sequential()
model.add(LSTM(units=64, input_shape=(seq_len, num_dims), return_sequences=True))
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
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 对测试集进行预测
y_pred = model.predict(X_test)

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

# 输出混淆矩阵
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)
print('Confusion matrix:\n', confusion_mtx)

num_train_samples = X_train.shape[0]
distances = np.zeros((X_test.shape[0], num_train_samples))
for i in range(X_test.shape[0]):
    for j in range(num_train_samples):
        distances[i, j] = np.linalg.norm(X_test[i] - X_train[j])
nearest_indices = np.argmin(distances, axis=1)
nearest_samples = X_train[nearest_indices]

# 输出结果
for i in range(X_test.shape[0]):
    print('Test sample:', X_test[i])
    print('Predicted class:', np.argmax(y_pred[i]))
    print('Nearest training sample:', nearest_samples[i])
    print('Distance:', distances[i, nearest_indices[i]])
    print()

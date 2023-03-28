import numpy as np
import pandas as pd
from keras.utils import to_categorical
from numpy import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2
import matplotlib.pyplot as plt


def normal():
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    data = np.vstack((voltage, current))
    return np.transpose(data)

def short_circuit_fault():
    # 短路故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    # 在第500个采样点模拟短路故障
    current[500:] = 0
    data = np.vstack((voltage, current))
    return np.transpose(data)


def open_circuit_fault():
    # 开路故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    # 在第500个采样点模拟开路故障
    current[500:] = 0
    voltage[500:] = 0
    data = np.vstack((voltage, current))
    return np.transpose(data)


def ground_fault():
    # 地闸故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    # 在第500个采样点模拟地闸故障
    current[500:] *= 10
    data = np.vstack((voltage, current))
    return np.transpose(data)


def insulation_fault():
    # 绝缘故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    # 在第500个采样点模拟绝缘故障
    voltage[500:] *= 0.5
    data = np.vstack((voltage, current))
    return np.transpose(data)


def overload_fault():
    # 过载故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    # 在第500个采样点模拟过载故障
    current[500:] *= 2
    data = np.vstack((voltage, current))
    return np.transpose(data)


def link_fault():
    # 链路故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    offset = random.randint(80)
    voltage = v * np.cos(2 * np.pi * t / 80 + offset)
    current = c * np.cos(2 * np.pi * t / 80 + offset)

    # 在第500个采样点模拟链路故障
    voltage[500:] *= 0.5
    current[500:] *= 0.5
    data = np.vstack((voltage, current))
    return np.transpose(data)


def build_x_train():
    train = []

    for i in range(100):
        train.append(normal())

    for i in range(100):
        train.append(short_circuit_fault())

    for i in range(100):
        train.append(open_circuit_fault())

    for i in range(100):
        train.append(ground_fault())

    for i in range(100):
        train.append(insulation_fault())

    for i in range(100):
        train.append(overload_fault())

    for i in range(100):
        train.append(link_fault())

    return np.stack(train, axis=0)


def build_y_train():
    return np.repeat(np.arange(7), 100)


def decode_one_hot(arr):
    result = []
    for i in range(len(arr)):
        result.append(np.argmax(arr[i]))
    return result


def encode_one_hot(arr, num):
    return to_categorical(arr, num_classes=num)


def load_x_test():
    file = 'raw_data/传感器厂家数据/电流互感单相数据.xls'
    data = pd.read_excel(file, usecols=['电流保护1', '电流计量数据'])
    data = data.to_numpy()
    # 对数据进行标准化
    scaler_1 = MinMaxScaler(feature_range=(-1, 1))
    scaler_2 = MinMaxScaler(feature_range=(-1.5, 1.5))

    data_x = scaler_1.fit_transform(data[:, 0].reshape(-1, 1)).flatten()
    data_y = scaler_2.fit_transform(data[:, 1].reshape(-1, 1)).flatten()
    data = np.vstack((data_y, data_x))
    data = np.transpose(data)
    data = data[:49000, :]
    data = data.reshape((49, 1000, 2))

    # 短路故障
    data[1, 500:, 1] = 0

    # 开路故障
    data[7, 500:, 0] = 0
    data[7, 500:, 1] = 0

    data[11, 500:, 0] = 0
    data[11, 500:, 1] = 0

    # 地闸故障
    data[19, 500:, 1] *= 10

    data[24, 500:, 1] *= 10

    # 绝缘故障
    data[31, 500:, 0] *= 0.5

    data[33, 500:, 0] *= 0.5

    # 过载故障
    data[34, 500:, 1] *= 2

    data[37, 500:, 1] *= 2

    # 链路故障
    data[45, 500:, 0] *= 0.5
    data[45, 500:, 1] *= 0.5

    data[48, 500:, 0] *= 0.5
    data[48, 500:, 1] *= 0.5

    return data


def load_y_test():
    data = np.zeros(49)
    data[1] = 1
    data[7] = 2
    data[11] = 2
    data[19] = 3
    data[24] = 3
    data[31] = 4
    data[33] = 4
    data[34] = 5
    data[37] = 5
    data[45] = 6
    data[48] = 6
    return data


x_train = build_x_train()

num_classes = 7
seq_len = x_train.shape[1]
num_dims = x_train.shape[2]
num_windows = 49
y_init = build_y_train()
y_train = encode_one_hot(y_init, num_classes)

x_test = load_x_test()
y_test = load_y_test()

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
model.fit(x_train, y_train, epochs=10, batch_size=32)



# 对测试集进行预测
t = np.linspace(0, 1000, 1000)
y_pred_one_hot = model.predict(x_test)
# 标签含义：0->正常, 1->短路故障, 2->开路故障, 3->地闸故障, 4->绝缘故障, 5->过载故障, 6->链路故障
y_pred = decode_one_hot(y_pred_one_hot)
label = ['normal', 'Short Circuit Fault', 'Open Circuit Fault', 'Ground Fault',
         'Insulation Fault', 'Overload Fault', 'Link Fault']
for i in range(num_windows):
    if y_pred[i] > 0:
        plt.plot(t, x_test[i, :, 0], label='l1')
        plt.plot(t, x_test[i, :, 1], label='l2')
        plt.legend()
        plt.title(label[y_pred[i]] + ' id:' + str(i))
        plt.show()

print(y_pred)
print('accuracy:', np.mean(y_pred == y_test))




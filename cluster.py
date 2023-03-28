import numpy as np
from numpy import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def normal():
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

    data = np.vstack((voltage, current))
    return np.transpose(data)

def short_circuit_fault():
    # 短路故障
    # 生成正常工作时的电压和电流数据
    t = np.linspace(0, 1000, 1000)
    v = random.uniform(1.0, 1.5)
    c = random.uniform(0.5, 1.0)
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

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
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

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
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

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
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

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
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

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
    voltage = v * np.cos(2 * np.pi * t / 80)
    current = c * np.cos(2 * np.pi * t / 80)

    # 在第500个采样点模拟链路故障
    voltage[500:] *= 0.5
    current[500:] *= 0.5
    data = np.vstack((voltage, current))
    return np.transpose(data)


def build_x_train():
    train = []

    for i in range(1000):
        train.append(normal())

    for i in range(10):
        train.append(short_circuit_fault())

    for i in range(10):
        train.append(open_circuit_fault())

    for i in range(10):
        train.append(ground_fault())

    for i in range(10):
        train.append(insulation_fault())

    for i in range(10):
        train.append(overload_fault())

    for i in range(10):
        train.append(link_fault())

    return np.stack(train, axis=0)


data = build_x_train()

# 对每个特征的每个时间步进行标准化处理
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

# 将数据转换为二维数组
X = data.reshape(data.shape[0], -1)

# 使用PCA将数据降维到2维
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# 使用高斯混合模型进行聚类
n_clusters = 3
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(X_pca)

# 计算每个样本点的异常得分
scores = gmm.score_samples(X_pca)

# 判断哪些数据是正常数据，哪些是故障数据
threshold = np.percentile(scores, 5)
normal_mask = scores >= threshold
normal_data = data[normal_mask]
abnormal_data = data[~normal_mask]

# 可视化正常数据和故障数据
fig, ax = plt.subplots()
ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], label='Normal data')
ax.scatter(X_pca[~normal_mask, 0], X_pca[~normal_mask, 1], label='Abnormal data')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.show()
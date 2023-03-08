import numpy as np
import pandas as pd
import scipy.signal as sig


# 时频域对齐
def domain_alignment():
    # 读入两个需要对齐的信号
    sig1 = np.loadtxt('signal1.txt')
    sig2 = np.loadtxt('signal2.txt')

    # 预处理信号
    # 在这个例子中，我们将信号进行高通滤波
    b, a = sig.butter(4, 0.1, 'highpass')
    sig1 = sig.filtfilt(b, a, sig1)
    sig2 = sig.filtfilt(b, a, sig2)

    # 调整滑动窗口大小和步长
    window_size = len(sig1) // 10
    step_size = window_size // 10

    # 计算互相关函数
    max_corr = 0
    max_shift = 0
    for shift in range(-window_size, window_size, step_size):
        corr = sig.correlate(sig1, sig2[shift:], mode='valid')
        if np.max(corr) > max_corr:
            max_corr = np.max(corr)
            max_shift = shift
    # 对第二个信号进行移位操作
    sig2_aligned = np.roll(sig2, max_shift)

    # 频域对齐
    sig1_fft = np.fft.fft(sig1)
    sig2_fft = np.fft.fft(sig2_aligned)

    # 计算相位差
    phase_diff = np.angle(sig2_fft) - np.angle(sig1_fft)

    # 将第二个信号的相位进行调整
    sig2_aligned = np.fft.ifft(np.abs(sig2_fft) * np.exp(1j * (np.angle(sig2_fft) - phase_diff)))
    return sig2_aligned


def pretreat():
    # 读取时序数据
    data = pd.read_csv('time_series_data.csv')

    # 清洗数据，去除空值和异常值
    data = data.dropna()
    data = data[(data['value'] > 0) & (data['value'] < 1000)]

    # 采样数据，将数据按小时采样
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    hourly_data = data.resample('H').mean()

    # 标准化数据，将数据缩放到0-1之间
    hourly_data = (hourly_data - hourly_data.min()) / (hourly_data.max() - hourly_data.min())
    return hourly_data


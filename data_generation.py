import numpy as np
import matplotlib.pyplot as plt

# 短路故障
# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltage = np.sin(2 * np.pi * 60 * t)
current = np.cos(2 * np.pi * 60 * t)

# 在第500个采样点模拟短路故障
current[500:] = 0

# 绘制电压和电流的时序图
plt.plot(t, voltage, label='Voltage')
plt.plot(t, current, label='Current')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Short Circuit Fault')
plt.show()


# 开路故障
# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltage = np.sin(2 * np.pi * 60 * t)
current = np.cos(2 * np.pi * 60 * t)

# 在第500个采样点模拟开路故障
current[500:] = 0
voltage[500:] = 0

# 绘制电压和电流的时序图
plt.plot(t, voltage, label='Voltage')
plt.plot(t, current, label='Current')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Open Circuit Fault')
plt.show()


# 地闸故障
# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltage = np.sin(2 * np.pi * 60 * t)
current = np.cos(2 * np.pi * 60 * t)

# 在第500个采样点模拟地闸故障
current[500:] *= 10

# 绘制电压和电流的时序图
plt.plot(t, voltage, label='Voltage')
plt.plot(t, current, label='Current')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Ground Fault')
plt.show()


# 绝缘故障
# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltage = np.sin(2 * np.pi * 60 * t)
current = np.cos(2 * np.pi * 60 * t)

# 在第500个采样点模拟绝缘故障
voltage[500:] *= 0.5

# 绘制电压和电流的时序图
plt.plot(t, voltage, label='Voltage')
plt.plot(t, current, label='Current')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Insulation Fault')
plt.show()


# 过载故障
# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltage = np.sin(2 * np.pi * 60 * t)
current = np.cos(2 * np.pi * 60 * t)

# 在第500个采样点模拟过载故障
current[500:] *= 2

# 绘制电压和电流的时序图
plt.plot(t, voltage, label='Voltage')
plt.plot(t, current, label='Current')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Overload Fault')
plt.show()


# 链路故障
# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltage = np.sin(2 * np.pi * 60 * t)
current = np.cos(2 * np.pi * 60 * t)

# 在第500个采样点模拟链路故障
voltage[500:] *= 0.5
current[500:] *= 0.5

# 绘制电压和电流的时序图
plt.plot(t, voltage, label='Voltage')
plt.plot(t, current, label='Current')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Link Fault')
plt.show()


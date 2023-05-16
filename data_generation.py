import numpy as np
import matplotlib.pyplot as plt


# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltageA = 10*np.sin(2 * np.pi * 50 * t)
currentA = 57*np.cos(2 * np.pi * 50 * t)
voltageB = 10*np.sin(2 * np.pi * 50 * t)
currentB = 57*np.cos(2 * np.pi * 50 * t)
voltageC = 10*np.sin(2 * np.pi * 50 * t)
currentC = 57*np.cos(2 * np.pi * 50 * t)

# 在第500个采样点模拟短路故障
currentA[500:] *= 10
voltageA[500:] /= 10

# 绘制电压和电流的时序图
plt.plot(t, voltageA, label='VoltageA')
plt.plot(t, currentA, label='CurrentA')
plt.plot(t, voltageB, label='VoltageB')
plt.plot(t, currentB, label='CurrentB')
plt.plot(t, voltageC, label='VoltageC')
plt.plot(t, currentC, label='CurrentC')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('kV/A')
plt.title('Neutral Grounding Fault')
plt.show()



# 生成正常工作时的电压和电流数据
t = np.linspace(0, 1, 1000)
voltageA = 10*np.sin(2 * np.pi * 50 * t)
currentA = 57*np.cos(2 * np.pi * 50 * t)
voltageB = 10*np.sin(2 * np.pi * 50 * t)
currentB = 57*np.cos(2 * np.pi * 50 * t)
voltageC = 10*np.sin(2 * np.pi * 50 * t)
currentC = 57*np.cos(2 * np.pi * 50 * t)

# 在第500个采样点模拟短路故障
currentA[500:] *= 3
voltageA[500:] *= 3
currentB[500:] /= 2
voltageB[500:] /= 2
currentC[500:] /= 2
voltageC[500:] /= 2

# 绘制电压和电流的时序图
plt.plot(t, voltageA, label='VoltageA')
plt.plot(t, currentA, label='CurrentA')
plt.plot(t, voltageB, label='VoltageB')
plt.plot(t, currentB, label='CurrentB')
plt.plot(t, voltageC, label='VoltageC')
plt.plot(t, currentC, label='CurrentC')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('kV/A')
plt.title('Neutral Grounding Fault')
plt.show()



# # 开路故障
# # 生成正常工作时的电压和电流数据
# t = np.linspace(0, 1, 1000)
# voltage = np.sin(2 * np.pi * 50 * t)
# current = np.cos(2 * np.pi * 50 * t)
#
# # 在第500个采样点模拟开路故障
# current[500:] = 0
# voltage[500:] = 0
#
# # 绘制电压和电流的时序图
# plt.plot(t, voltage, label='Voltage')
# plt.plot(t, current, label='Current')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Open Circuit Fault')
# plt.show()

#
# # 地闸故障
# # 生成正常工作时的电压和电流数据
# t = np.linspace(0, 1, 1000)
# voltage = np.sin(2 * np.pi * 60 * t)
# current = np.cos(2 * np.pi * 60 * t)
#
# # 在第500个采样点模拟地闸故障
# current[500:] *= 10
#
# # 绘制电压和电流的时序图
# plt.plot(t, voltage, label='Voltage')
# plt.plot(t, current, label='Current')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Ground Fault')
# plt.show()
#
#
# # 绝缘故障
# # 生成正常工作时的电压和电流数据
# t = np.linspace(0, 1, 1000)
# voltage = np.sin(2 * np.pi * 60 * t)
# current = np.cos(2 * np.pi * 60 * t)
#
# # 在第500个采样点模拟绝缘故障
# voltage[500:] *= 0.5
#
# # 绘制电压和电流的时序图
# plt.plot(t, voltage, label='Voltage')
# plt.plot(t, current, label='Current')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Insulation Fault')
# plt.show()
#
#
# # 过载故障
# # 生成正常工作时的电压和电流数据
# t = np.linspace(0, 1, 1000)
# voltage = np.sin(2 * np.pi * 60 * t)
# current = np.cos(2 * np.pi * 60 * t)
#
# # 在第500个采样点模拟过载故障
# current[500:] *= 2
#
# # 绘制电压和电流的时序图
# plt.plot(t, voltage, label='Voltage')
# plt.plot(t, current, label='Current')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Overload Fault')
# plt.show()
#
#
# # 链路故障
# # 生成正常工作时的电压和电流数据
# t = np.linspace(0, 1, 1000)
# voltage = np.sin(2 * np.pi * 60 * t)
# current = np.cos(2 * np.pi * 60 * t)
#
# # 在第500个采样点模拟链路故障
# voltage[500:] *= 0.5
# current[500:] *= 0.5
#
# # 绘制电压和电流的时序图
# plt.plot(t, voltage, label='Voltage')
# plt.plot(t, current, label='Current')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Link Fault')
# plt.show()
#

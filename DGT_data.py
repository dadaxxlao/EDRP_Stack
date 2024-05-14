import matplotlib.pyplot as plt
import numpy as np


# 定义数据
t = np.linspace(0, 100, 500)  # 时间轴，0到100毫秒，500个数据点
tcp = 70 * (1 - np.exp(-t / 16))  # TCP数据，上凸曲线，增长到70%
udp = 26 * t / 48
udp = udp[t <= 48]  # 仅在t<=48ms时绘制UDP数据
t_udp = t[t <= 48]  # 对应的时间数据


# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(t_udp, udp, label="UDP", linestyle='-.', color='#8ECFC9')
plt.plot(t, tcp, label="TCP", linestyle='-.', color='#FFBE7A')

plt.axhline(y=30, color='#f8ac8c', linestyle='--')
plt.axhline(y=70, color='#c82423', linestyle='--')
plt.axvline(x=48,ymax=70/100, color='#ff8884', linestyle='--')

plt.fill_between(t, tcp, 70, where=(tcp < 70) & (t > 48), color='gray', alpha=0.5, label='Long Tail Flow')

np.random.seed(0)  # 保持随机点可复现
sample_indices = np.random.choice(t.size, 15, replace=False)  # 从时间数组中随机选择十个点
sample_indices_udp = np.random.choice(t_udp.size, 15, replace=False)  # 对UDP做同样操作

plt.scatter(t[sample_indices], tcp[sample_indices] + np.random.normal(0, 1.4, 15), color='#FFBE7A', alpha=1, s=30, label="TCP Data Points")
plt.scatter(t_udp[sample_indices_udp], udp[sample_indices_udp] + np.random.normal(0, 1.3, 15), color='#8ECFC9', alpha=1, s=30, label="UDP Data Points")


# 设置图例
plt.legend()
plt.ylim(0, 100)
plt.xlim(0, 100)

# 添加标题和轴标签
plt.title("Flow Completion Partition Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("Flow Completion Partition (%)")

# 显示图形
plt.show()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec


time_str = sys.argv[1]

os.chdir(os.path.dirname(__file__))
os.chdir("../cache/saved")


record_power_A = np.load(f"{time_str}_record_power_A.npy")
record_power_B = np.load(f"{time_str}_record_power_B.npy")
record_waveform_A = np.load(f"{time_str}_record_waveform_A.npy")
record_waveform_B = np.load(f"{time_str}_record_waveform_B.npy")
record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)
# time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval = np.load(f"{time_str}_parameters.npy", allow_pickle=True)
data_dict = np.load(f"{time_str}_parameters_dict.npy", allow_pickle=True).item()
f_A = data_dict["f_A"]
f_B = data_dict["f_B"]
J_back_r = data_dict["J_back_r"]
mode_number = data_dict["mode_number"]
zeta_ini = data_dict["zeta_ini"]
zeta_end = data_dict["zeta_end"]
d_2 = data_dict["d_2"]
xs_freq = np.arange(-mode_number / 2, mode_number / 2)
zetas = np.load(f"{time_str}_zetas.npy")
length_zetas = len(zetas)
length = len(record_waveform_A)
print(f"Valid range: [0, {length})")

# 创建图形和网格布局
# plt.ion()
fig = plt.figure(figsize=(8, 7))  # 调整图像大小以腾出更多空间
gs = GridSpec(2, 2, height_ratios=[1, 0.7])  # 第一、二幅图各占1/3，第三幅图占1/2

# 创建轴
ax_wave = fig.add_subplot(gs[0, 0])  # 第一幅图
ax_freq = fig.add_subplot(gs[0, 1])        # 第二幅图
ax_power = fig.add_subplot(gs[1, :])        # 功率图

# 绘制功率图
ax_power.plot(zetas, record_power_A, label=f'f_A = {f_A}', alpha=0.7)
ax_power.plot(zetas, record_power_B, label=f'f_B = {f_B}', alpha=0.7)
ax_power.set_xlim(zeta_ini, zeta_end)
ax_power.set_title(f"Power, J = {J_back_r}, d_2 = {d_2}")
ax_power.legend(loc="lower left", fontsize=8)
# 在功率图中添加竖线
detuning_line = ax_power.axvline(x=zetas[0], color='red', linestyle='--')
ax_power.legend(loc="lower left", fontsize='small')

# 创建滑块的子图 (位于底部，水平居中)
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
slider = Slider(ax_slider, 'Iteration', 0, length-1, valinit=0, valstep=1)

# 创建加减按钮的位置 (放在滑块下方)
ax_button_minus = plt.axes([0.39, 0.01, 0.1, 0.04])  # 左侧按钮
ax_button_plus = plt.axes([0.51, 0.01, 0.1, 0.04])   # 右侧按钮

button_minus = Button(ax_button_minus, '-')
button_plus = Button(ax_button_plus, '+')

# 定义按钮点击后的行为
def decrease(event):
    current_val = slider.val
    slider.set_val(max(0, current_val - 1))  # 减小滑块值

def increase(event):
    current_val = slider.val
    slider.set_val(min(length - 1, current_val + 1))  # 增加滑块值

button_minus.on_clicked(decrease)
button_plus.on_clicked(increase)

ax_wave.set_ylim(0, max(np.abs(record_waveform_A).max(), np.abs(record_waveform_B).max()))
ax_freq.set_yscale('log')
ax_freq.set_ylim(0.3 * min(np.abs(record_freq_A[-1])[0], np.abs(record_freq_B[-1])[0]), max(np.abs(record_freq_A).max(), np.abs(record_freq_B).max()))

line_A, = ax_wave.plot(np.abs(record_waveform_A[0]), alpha=0.7)
line_B, = ax_wave.plot(np.abs(record_waveform_B[0]), alpha=0.7)
ax_wave.set_title(f"Iteration: {0}, detuning: {zetas[0]:.2f}")
line_A_freq, = ax_freq.plot(xs_freq, np.abs(record_freq_A[0]), alpha=0.7)
line_B_freq, = ax_freq.plot(xs_freq, np.abs(record_freq_B[0]), alpha=0.7)

def update(val):
    iter = int(slider.val)
    detuning = zetas[iter * length_zetas // length]
    ax_wave.set_title(f"Iteration: {iter}, detuning: {detuning:.2f}")
    line_A.set_ydata(np.abs(record_waveform_A[iter]))
    line_B.set_ydata(np.abs(record_waveform_B[iter]))
    line_A_freq.set_ydata(np.abs(record_freq_A[iter]))
    line_B_freq.set_ydata(np.abs(record_freq_B[iter]))
    detuning_line.set_xdata([detuning, detuning])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.subplots_adjust(hspace=0.3, bottom=0.16)
plt.show()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

time_str = sys.argv[1]

os.chdir(os.path.dirname(__file__))
os.chdir("../cache/saved")

record_waveform_A = np.load(f"{time_str}_record_waveform_A.npy")
record_waveform_B = np.load(f"{time_str}_record_waveform_B.npy")
record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)
time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval = np.load(f"{time_str}_parameters.npy", allow_pickle=True)
xs_freq = np.arange(-mode_number / 2, mode_number / 2)
zetas = np.load(f"{time_str}_zetas.npy")

print("length of zetas:", len(zetas))
print("length of record_waveform:", len(record_waveform_A))


plt.ion()
fig, axs = plt.subplots(2)
fig.set_size_inches(5, 7)
ax, ax_freq = axs[0], axs[1]

ax.set_ylim(0, max(np.abs(record_waveform_A).max(), np.abs(record_waveform_B).max()))
ax_freq.set_yscale('log')
ax_freq.set_ylim(0.3 * min(np.abs(record_freq_A[-1])[0], np.abs(record_freq_B[-1])[0]), max(np.abs(record_freq_A).max(), np.abs(record_freq_B).max()))

line_A, = ax.plot(np.abs(record_waveform_A[0]), alpha = 0.7)
line_B, = ax.plot(np.abs(record_waveform_B[0]), alpha = 0.7)
line_A_freq, = ax_freq.plot(xs_freq, np.abs(record_freq_A[0]), alpha = 0.7)
line_B_freq, = ax_freq.plot(xs_freq, np.abs(record_freq_B[0]), alpha = 0.7)
# plt.show()

length = len(record_waveform_A)
print(f"Valid range: [0, {length})")


def play():
    for i in range(length):
        line_A.set_ydata(np.abs(record_waveform_A[i]))
        line_B.set_ydata(np.abs(record_waveform_B[i]))
        line_A_freq.set_ydata(np.abs(record_freq_A[i]))
        line_B_freq.set_ydata(np.abs(record_freq_B[i]))
        detuning = zetas[i * power_interval]
        plt.title(f"Iteration: {i}, detuning: {detuning:.2f}")
        plt.pause(0.00001)


while True:
    iter = input("Enter the iteration number, or 'exit' to exit: ")
    if iter == "exit":
        break
    if iter == "play":
        play()
        continue
    try:
        iter = int(iter)
    except:
        print("Invalid input")
        continue
    if not 0 <= int(iter) < length:
        print("Iteration number out of range")
        print(f"Valid range: [0, {length})")
        continue
    line_A.set_ydata(np.abs(record_waveform_A[iter]))
    line_B.set_ydata(np.abs(record_waveform_B[iter]))
    line_A_freq.set_ydata(np.abs(record_freq_A[iter]))
    line_B_freq.set_ydata(np.abs(record_freq_B[iter]))
    detuning = zetas[iter * power_interval]
    plt.title(f"Iteration: {iter}, detuning: {detuning:.2f}")
    plt.draw()


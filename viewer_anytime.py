import os
import sys
import numpy as np
import matplotlib.pyplot as plt



time_str = sys.argv[1]

os.chdir(os.path.dirname(__file__))
os.chdir("../cache/saved")

# record_power_A = np.load(f"{time_str}_record_power_A.npy")
# record_power_B = np.load(f"{time_str}_record_power_B.npy")
record_waveform_A = np.load(f"{time_str}_record_waveform_A.npy")
record_waveform_B = np.load(f"{time_str}_record_waveform_B.npy")
record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)
time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval = np.load(f"{time_str}_parameters.npy", allow_pickle=True)
zetas = np.load(f"{time_str}_zetas.npy")
print("length of zetas:", len(zetas))
print("length of record_waveform:", len(record_waveform_A))


plt.ion()
fig, axs = plt.subplots(2)
fig.set_size_inches(5, 7)
ax, ax_freq = axs[0], axs[1]
line_A, = ax.plot(np.abs(record_waveform_A[-1]), alpha = 0.7)
line_B, = ax.plot(np.abs(record_waveform_B[-1]), alpha = 0.7)
line_A_freq, = ax_freq.plot(np.abs(record_freq_A[-1]), alpha = 0.7)
line_B_freq, = ax_freq.plot(np.abs(record_freq_B[-1]), alpha = 0.7)
# plt.show()

length = len(record_waveform_A)
print(f"Valid range: [0, {length})")

while True:
    iter = input("Enter the iteration number, or 'exit' to exit: ")
    if iter == "exit":
        break
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
    ax.set_ylim(0, 1.1 * max(np.abs(record_waveform_A[iter]).max(), np.abs(record_waveform_B[iter]).max()))
    line_A_freq.set_ydata(np.abs(record_freq_A[iter]))
    line_B_freq.set_ydata(np.abs(record_freq_B[iter]))
    ax_freq.set_ylim(0, 1.1 * max(np.abs(record_freq_A[iter]).max(), np.abs(record_freq_B[iter]).max()))
    detuning = zetas[iter]
    plt.title(f"Iteration: {iter}, detuning: {detuning:.2f}")
    plt.draw()

    


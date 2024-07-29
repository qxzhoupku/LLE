import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# import sys
# import time

# create a folder to save the outputs
# first, get the path of the current file
current_path = os.path.abspath(__file__)
print(current_path)
# then, change the directory to the parent directory
os.chdir(os.path.dirname(current_path))
if not os.path.exists("../output"):
    os.mkdir("../output")
os.chdir("../output")


from parameters import mode_number, iter_number, plot_interval, record_interval, zeta_ini, zeta_end, zetas, f_A, f_B, J_back_r, loss_back, r_back, t_back, delta_t, D_int, time_str, plot_flag # type: ignore


# plt.plot(D_int)
# plt.show()

def noise(mode_number):
    return np.random.random(mode_number) * np.exp(1j * np.random.random(mode_number)) / 2


# create np.array, which length is mode_number, element of A is complex
A_freq = noise(mode_number)
B_freq = noise(mode_number)
A = np.fft.ifft(np.fft.ifftshift(A_freq))
B = np.fft.ifft(np.fft.ifftshift(B_freq))

if plot_flag == True:
    plt.ion()
    fig, axs = plt.subplots(2)
    ax, ax_freq = axs[0], axs[1]
    y_max, y_freq_max = np.max(np.abs(A)), np.max(np.abs(A_freq))
    ax.set_ylim(0, 1.2 * y_max)
    line_A, = ax.plot(np.abs(A))
    line_B, = ax.plot(np.abs(B))
    ax_freq.set_ylim(0, 1.2 * y_freq_max)
    ax_freq.set_yscale('log')
    xs_freq = np.arange(-mode_number / 2, mode_number / 2)
    line_A_freq, = ax_freq.plot(xs_freq, np.abs(A_freq))
    line_B_freq, = ax_freq.plot(xs_freq, np.abs(B_freq))



# fft = lambda x: np.fft.fftshift(np.fft.fft(x))
# ifft = lambda x: np.fft.ifft(np.fft.ifftshift(x))

def cal_power(x):
    mode_number = len(x)
    return np.sum(np.abs(x)**2) / mode_number


def split_step(A_0, zeta, f, D_int, delta_t, B, J_back_r = 0, r_back = 0):
    B_avg_pow = cal_power(B)
    A_1 = np.exp(1j * (np.abs(A_0)**2 + B_avg_pow) * delta_t) * A_0
    A_1_freq = np.fft.fftshift(np.fft.fft(A_1))
    A_2_freq = np.exp(-(1 + J_back_r + 1j * zeta + 1j * D_int) * delta_t) * A_1_freq
    # A_2_freq += noise(mode_number) * delta_t * 10
    A_2 = np.fft.ifft(np.fft.ifftshift(A_2_freq))
    A_3 = A_2 + f * delta_t
    # backscattering term, in and out
    A_4 = A_3 + 1j * J_back_r * delta_t * B[::-1]
    return A_4


def figure_plot(A, B, i):
    global y_max, y_freq_max

    line_A.set_ydata(np.abs(A))
    line_B.set_ydata(np.abs(B))
    y_max = np.max([np.max(np.abs(A)), y_max])
    ax.set_ylim(0, 1.2 * y_max)
    ax.title.set_text(f"zeta = %.2f, proc = %.2f%%" % (zeta, i / iter_number * 100))

    A_freq = np.fft.fftshift(np.fft.fft(A))
    B_freq = np.fft.fftshift(np.fft.fft(B))
    line_A_freq.set_ydata(np.abs(A_freq))
    line_B_freq.set_ydata(np.abs(B_freq))
    # y_freq_max = np.max([np.max(np.abs(A_freq)), y_freq_max])
    y_freq_max = np.max(np.abs(A_freq))
    ax_freq.set_ylim(np.min(np.abs(A_freq)), 1.2 * y_freq_max)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.pause(0.0001)
    return



record_power_A = np.zeros(iter_number)
record_power_B = np.zeros(iter_number)
record_waveform_A = \
    np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)
record_waveform_B = \
    np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)


print("Start main loop")
################
# Main loop
# Add a progress bar
# for i in range(iter_number):
for i in tqdm(range(iter_number), desc="Processing"):
    zeta = zetas[i]
    A_new = split_step(A, zeta, f_A, D_int, delta_t, B, J_back_r, r_back)
    B_new = split_step(B, zeta, f_B, D_int, delta_t, A, J_back_r, r_back)
    A, B = A_new, B_new
    record_power_A[i] = cal_power(A)
    record_power_B[i] = cal_power(B)

    if i % record_interval == 0:
        record_waveform_A[i // record_interval] = A
        record_waveform_B[i // record_interval] = B


    if i % plot_interval == 0 and plot_flag == True:
        figure_plot(A, B, i)
################
print("End main loop")
plt.ioff()


# store D_int
np.savetxt(f"{time_str}_D_int.txt", D_int)


# plot the power
plt.figure()
plt.plot(zetas, record_power_A)
plt.plot(zetas, record_power_B)
plt.xlim(zeta_ini, zeta_end)
plt.savefig(f"{time_str}_power.png", dpi=600)

# plot the waveform in a heatmap
record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)
record_freq_A = record_freq_A.T
record_freq_B = record_freq_B.T
record_waveform_A = record_waveform_A.T
record_waveform_B = record_waveform_B.T

plt.figure()
plt.imshow(np.abs(record_waveform_A), aspect='auto', \
           extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Waveform_A")
plt.savefig(f"{time_str}_waveform_A.png", dpi=600)
print("Waveform_A saved")

plt.figure()
plt.imshow(np.abs(record_waveform_B), aspect='auto', \
           extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Waveform_B")
plt.savefig(f"{time_str}_waveform_B.png", dpi=600)
print("Waveform_B saved")

# plot the frequency in a heatmap
plt.figure()
plt.imshow(np.abs(record_freq_A), aspect='auto', \
           extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Frequency_A")
plt.savefig(f"{time_str}_frequency_A.png", dpi=600)
print("Frequency_A saved")

plt.figure()
plt.imshow(np.abs(record_freq_B), aspect='auto', \
           extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Frequency_B")
plt.savefig(f"{time_str}_frequency_B.png", dpi=600)
print("Frequency_B saved")

# plt.show()




